/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License version 2 as
* published by the Free Software Foundation;
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#undef PGO_TRAINING
#define PATH_TO_PGO_CONFIG "path_to_pgo_config"

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <set>
#include <time.h> 
#include "ns3/core-module.h"
#include "ns3/qbb-helper.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"
#include "ns3/global-route-manager.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/packet.h"
#include "ns3/error-model.h"
#include <ns3/rdma.h>
#include <ns3/rdma-client.h>
#include <ns3/rdma-client-helper.h>
#include <ns3/rdma-driver.h>
#include <ns3/switch-node.h>
#include <ns3/sim-setting.h>

using namespace ns3;
using namespace std;

NS_LOG_COMPONENT_DEFINE("GENERIC_SIMULATION");

// global static configurations
uint32_t cc_mode = 1;
bool enable_qcn = true, use_dynamic_pfc_threshold = true;
uint32_t packet_payload_size = 1000, l2_chunk_size = 0, l2_ack_interval = 0;
double pause_time = 5, simulator_stop_time = 3.01;
std::string topology_file, flow_file;
std::string fct_output_file = "fct.txt";
std::string pfc_output_file = "pfc.txt";
std::string path_output_file = "path.txt";
std::string config_output_file = "config.txt";

// variable configurations 
struct hostConfig{
	uint32_t ewma_shift;
	double alpha_resume_interval = 55, rp_timer, ewma_gain = 1 / 16;
	double rate_decrease_interval = 4;
};
hostConfig hConf[1344];

struct SwitchConfig{
	uint64_t kmax;
	uint32_t kmin;
	double pmax;
	int32_t alpha;
};
SwitchConfig swConf[1344];

uint32_t fast_recovery_times = 5;
std::string rate_ai, rate_hai, min_rate = "100Mb/s";
std::string dctcp_rate_ai = "1000Mb/s";

bool clamp_target_rate = false, l2_back_to_zero = false;
double error_rate_per_link = 0.0;
uint32_t has_win = 1;
uint32_t global_t = 1;
uint32_t mi_thresh = 5;
bool var_win = false, fast_react = true;
bool multi_rate = true;
bool sample_feedback = false;
double pint_log_base = 1.05;
double pint_prob = 1.0;
double u_target = 0.95;
uint32_t int_multi = 1;
bool rate_bound = true;

uint32_t ack_high_prio = 0;
uint64_t link_down_time = 0;
uint32_t link_down_A = 0, link_down_B = 0;

uint32_t buffer_size = 16;

uint32_t qlen_dump_interval = 100000000, qlen_mon_interval = 100;
uint64_t qlen_mon_start = 2000000000, qlen_mon_end = 2100000000;
std::string qlen_mon_file;

unordered_map<uint64_t, uint32_t> rate2kmax, rate2kmin;
unordered_map<uint64_t, double> rate2pmax;

/************************************************
 * Runtime varibles
 ***********************************************/
std::ifstream topof, flowf;
uint32_t node_num;

NodeContainer n;
uint64_t nic_rate;
uint64_t maxRtt, maxBdp;

struct Interface{
	uint32_t idx;
	bool up;
	uint64_t delay;
	uint64_t bw;

	Interface() : idx(0), up(false){}
};
map<Ptr<Node>, map<Ptr<Node>, Interface> > nbr2if;
// Mapping destination to next hop for each node: <node, <dest, <nexthop0, ...> > >
map<Ptr<Node>, map<Ptr<Node>, vector<Ptr<Node> > > > nextHop;
map<Ptr<Node>, map<Ptr<Node>, uint64_t> > pairDelay;
map<Ptr<Node>, map<Ptr<Node>, uint64_t> > pairTxDelay;
map<uint32_t, map<uint32_t, uint64_t> > pairBw;
map<Ptr<Node>, map<Ptr<Node>, uint64_t> > pairBdp;
map<uint32_t, map<uint32_t, uint64_t> > pairRtt;

std::vector<Ipv4Address> serverAddress;

// maintain port number for each host pair
std::unordered_map<uint32_t, unordered_map<uint32_t, uint16_t> > portNumder;

struct FlowInput{
	uint32_t src, dst, pg, maxPacketCount, port, dport;
	double start_time;
	uint32_t idx;
};
FlowInput flow_input = {0};
uint32_t flow_num;

map<uint32_t, map<uint32_t, map<uint32_t, uint32_t>>> flowMap;
std::unordered_map<uint32_t, std::vector<uint32_t> > flowPath;
std::unordered_map<uint32_t, std::vector<uint32_t> > flowNode;
std::unordered_map<uint32_t, std::set<uint32_t> > linkInFlow;
std::unordered_map<uint32_t, std::set<uint32_t> > nodeInFlow;
std::unordered_map<uint32_t, map<uint32_t, uint64_t> > pktStartTime;
std::unordered_map<uint32_t, map<uint64_t, uint32_t>> pktDelay;
std::unordered_map<uint32_t, std::set<uint32_t> > nodeLink;
std::vector<bool> linkActivated;
uint32_t flowFinished = 0;

bool logPath = true;

void ReadFlowInput(){
	if (flow_input.idx < flow_num){
		flowf >> flow_input.src >> flow_input.dst >> flow_input.pg >> flow_input.dport >> flow_input.maxPacketCount >> flow_input.start_time;
		NS_ASSERT(n.Get(flow_input.src)->GetNodeType() == 0 && n.Get(flow_input.dst)->GetNodeType() == 0);
	}
}
void ScheduleFlowInputs(){
	while (flow_input.idx < flow_num && Seconds(flow_input.start_time) == Simulator::Now()){
		uint32_t port = portNumder[flow_input.src][flow_input.dst]++; // get a new port number 
		RdmaClientHelper clientHelper(flow_input.pg, serverAddress[flow_input.src], serverAddress[flow_input.dst], port, flow_input.dport, flow_input.maxPacketCount, has_win?(global_t==1?maxBdp:pairBdp[n.Get(flow_input.src)][n.Get(flow_input.dst)]):0, global_t==1?maxRtt:pairRtt[flow_input.src][flow_input.dst]);
		ApplicationContainer appCon = clientHelper.Install(n.Get(flow_input.src));
		appCon.Start(Time(0));
		
		flowMap[flow_input.src][flow_input.dst][port] = flow_input.idx;
		linkInFlow[flow_input.idx].insert(flow_input.src);
		flowPath[flow_input.idx].push_back(flow_input.src);
		nodeInFlow[flow_input.idx].insert(flow_input.src);
		// flowNode[flow_input.idx].push_back(flow_input.src);
		// get the next flow input
		flow_input.idx++;
		ReadFlowInput();
	}

	// schedule the next time to run this function
	if (flow_input.idx < flow_num){
		Simulator::Schedule(Seconds(flow_input.start_time)-Simulator::Now(), ScheduleFlowInputs);
	}else { // no more flows, close the file
		flowf.close();
	}
}

Ipv4Address node_id_to_ip(uint32_t id){
	return Ipv4Address(0x0b000001 + ((id / 256) * 0x00010000) + ((id % 256) * 0x00000100));
}

uint32_t ip_to_node_id(Ipv4Address ip){
	return (ip.Get() >> 8) & 0xffff;
}

void qp_finish(FILE* fout, Ptr<RdmaQueuePair> q){
	uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
	uint64_t base_rtt = pairRtt[sid][did], b = pairBw[sid][did];
	uint32_t total_bytes = q->m_size + ((q->m_size-1) / packet_payload_size + 1) * (CustomHeader::GetStaticWholeHeaderSize() - IntHeader::GetStaticSize()); // translate to the minimum bytes required (with header but no INT)
	uint64_t standalone_fct = base_rtt + total_bytes * 8000000000lu / b;
	// sip, dip, sport, dport, size (B), start_time, fct (ns), standalone_fct (ns)
	// fprintf(fout, "%u %u %u %u %lu %lu %lu %lu\n", sid, did, q->sport, q->dport, q->m_size, q->startTime.GetTimeStep(), (Simulator::Now() - q->startTime).GetTimeStep(), standalone_fct);
	uint32_t fid = flowMap[sid][did][q->sport];
	fprintf(fout, "%u,%lu,%u,%u,%lu,%lu,%lu\n", fid, q->m_size, sid, did, q->startTime.GetTimeStep(), (Simulator::Now() - q->startTime).GetTimeStep(), standalone_fct);
	fflush(fout);
	flowFinished++;
	if(flowFinished == flow_num){
		Simulator::Stop();
	}
}

void get_pfc(FILE* fout, Ptr<QbbNetDevice> dev, uint32_t type){
	fprintf(fout, "%lu %u %u %u %u\n", Simulator::Now().GetTimeStep(), dev->GetNode()->GetId(), dev->GetNode()->GetNodeType(), dev->GetIfIndex(), type);
}

struct QlenDistribution{
	vector<uint32_t> cnt; // cnt[i] is the number of times that the queue len is i KB

	void add(uint32_t qlen){
		uint32_t kb = qlen / 1000;
		if (cnt.size() < kb+1)
			cnt.resize(kb+1);
		cnt[kb]++;
	}
};

map<uint32_t, map<uint32_t, QlenDistribution> > queue_result;
void monitor_buffer(FILE* qlen_output, NodeContainer *n){
	for (uint32_t i = 0; i < n->GetN(); i++){
		if (n->Get(i)->GetNodeType() == 1){ // is switch
			Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(n->Get(i));
			if (queue_result.find(i) == queue_result.end())
				queue_result[i];
			for (uint32_t j = 1; j < sw->GetNDevices(); j++){
				uint32_t size = 0;
				for (uint32_t k = 0; k < SwitchMmu::qCnt; k++)
					size += sw->m_mmu->egress_bytes[j][k];
				queue_result[i][j].add(size);
			}
		}
	}

	if (Simulator::Now().GetTimeStep() % qlen_dump_interval == 0){
		fprintf(qlen_output, "time: %lu\n", Simulator::Now().GetTimeStep());
		for (auto &it0 : queue_result)
		{
			for (auto &it1 : it0.second){
				fprintf(qlen_output, "%u %u", it0.first, it1.first);
				auto &dist = it1.second.cnt;
				for (uint32_t i = 0; i < dist.size(); i++)
					fprintf(qlen_output, " %u", dist[i]);
				fprintf(qlen_output, "\n");
			}
		}
		fflush(qlen_output);
	}

	if (Simulator::Now().GetTimeStep() < qlen_mon_end)
		Simulator::Schedule(NanoSeconds(qlen_mon_interval), &monitor_buffer, qlen_output, n);
	
}


void CalculateRoute(Ptr<Node> host){
	// queue for the BFS.
	vector<Ptr<Node> > q;
	// Distance from the host to each node.
	map<Ptr<Node>, int> dis;
	map<Ptr<Node>, uint64_t> delay;
	map<Ptr<Node>, uint64_t> txDelay;
	map<Ptr<Node>, uint64_t> bw;
	// init BFS.
	q.push_back(host);
	dis[host] = 0;
	delay[host] = 0;
	txDelay[host] = 0;
	bw[host] = 0xfffffffffffffffflu;
	// BFS.
	for (int i = 0; i < (int)q.size(); i++){
		Ptr<Node> now = q[i];
		int d = dis[now];
		for (auto it = nbr2if[now].begin(); it != nbr2if[now].end(); it++){
			// skip down link
			if (!it->second.up)
				continue;
			Ptr<Node> next = it->first;
			// If 'next' have not been visited.
			if (dis.find(next) == dis.end()){
				dis[next] = d + 1;
				delay[next] = delay[now] + it->second.delay;
				txDelay[next] = txDelay[now] + packet_payload_size * 1000000000lu * 8 / it->second.bw;
				bw[next] = std::min(bw[now], it->second.bw);
				// we only enqueue switch, because we do not want packets to go through host as middle point
				if (next->GetNodeType() == 1)
					q.push_back(next);
			}
			// if 'now' is on the shortest path from 'next' to 'host'.
			if (d + 1 == dis[next]){
				nextHop[next][host].push_back(now);
			}
		}
	}
	for (auto it : delay)
		pairDelay[it.first][host] = it.second;
	for (auto it : txDelay)
		pairTxDelay[it.first][host] = it.second;
	for (auto it : bw)
		pairBw[it.first->GetId()][host->GetId()] = it.second;
}

void CalculateRoutes(NodeContainer &n){
	for (int i = 0; i < (int)n.GetN(); i++){
		Ptr<Node> node = n.Get(i);
		if (node->GetNodeType() == 0)
			CalculateRoute(node);
	}
}

void SetRoutingEntries(){
	// For each node.
	for (auto i = nextHop.begin(); i != nextHop.end(); i++){
		Ptr<Node> node = i->first;
		auto &table = i->second;
		for (auto j = table.begin(); j != table.end(); j++){
			// The destination node.
			Ptr<Node> dst = j->first;
			// The IP address of the dst.
			Ipv4Address dstAddr = dst->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
			// The next hops towards the dst.
			vector<Ptr<Node> > nexts = j->second;
			for (int k = 0; k < (int)nexts.size(); k++){
				Ptr<Node> next = nexts[k];
				uint32_t interface = nbr2if[node][next].idx;
				if (node->GetNodeType() == 1)
					DynamicCast<SwitchNode>(node)->AddTableEntry(dstAddr, interface);
				else{
					node->GetObject<RdmaDriver>()->m_rdma->AddTableEntry(dstAddr, interface);
				}
			}
		}
	}
}

// take down the link between a and b, and redo the routing
void TakeDownLink(NodeContainer n, Ptr<Node> a, Ptr<Node> b){
	if (!nbr2if[a][b].up)
		return;
	// take down link between a and b
	nbr2if[a][b].up = nbr2if[b][a].up = false;
	nextHop.clear();
	CalculateRoutes(n);
	// clear routing tables
	for (uint32_t i = 0; i < n.GetN(); i++){
		if (n.Get(i)->GetNodeType() == 1)
			DynamicCast<SwitchNode>(n.Get(i))->ClearTable();
		else
			n.Get(i)->GetObject<RdmaDriver>()->m_rdma->ClearTable();
	}
	DynamicCast<QbbNetDevice>(a->GetDevice(nbr2if[a][b].idx))->TakeDown();
	DynamicCast<QbbNetDevice>(b->GetDevice(nbr2if[b][a].idx))->TakeDown();
	// reset routing table
	SetRoutingEntries();

	// redistribute qp on each host
	for (uint32_t i = 0; i < n.GetN(); i++){
		if (n.Get(i)->GetNodeType() == 0)
			n.Get(i)->GetObject<RdmaDriver>()->m_rdma->RedistributeQp();
	}
}

uint64_t get_nic_rate(NodeContainer &n){
	for (uint32_t i = 0; i < n.GetN(); i++){
		if (n.Get(i)->GetNodeType() == 0){
			return DynamicCast<QbbNetDevice>(n.Get(i)->GetDevice(1))->GetDataRate().GetBitRate();
		}
	}	
	return 0;
}

void qbb_enqueue(uint32_t i, Ptr<QbbNetDevice> dev, CustomHeader ch)
{
	uint32_t src_id = ip_to_node_id(Ipv4Address(ch.sip));
	uint32_t dst_id = ip_to_node_id(Ipv4Address(ch.dip));
	uint32_t port_id = ch.udp.sport;
	uint32_t flow_id = flowMap[src_id][dst_id][port_id];
	uint32_t node_id = dev->GetNode()->GetId();
	if(port_id >= 10000)
	{
		if(linkInFlow[flow_id].find(i) == linkInFlow[flow_id].end())
		{
			linkInFlow[flow_id].insert(i);
			flowPath[flow_id].push_back(i);
		}
		if(nodeInFlow[flow_id].find(node_id) == nodeInFlow[flow_id].end())
		{
			nodeInFlow[flow_id].insert(node_id);
			flowNode[flow_id].push_back(node_id);
		}
	} 
}

void host_send(uint32_t i, Ptr<const Packet> pkt)
{
	CustomHeader ch(CustomHeader::L2_Header | CustomHeader::L3_Header | CustomHeader::L4_Header);
	pkt->PeekHeader(ch);
	uint32_t src_id = ip_to_node_id(Ipv4Address(ch.sip));
	uint32_t dst_id = ip_to_node_id(Ipv4Address(ch.dip));
	uint32_t port_id = ch.udp.sport;
	uint32_t flow_id = flowMap[src_id][dst_id][port_id];
	uint32_t pkt_seq = ch.udp.seq / packet_payload_size;
	pktStartTime[flow_id][pkt_seq] =  Simulator::Now().GetNanoSeconds();
}

void host_receive(uint32_t i, Ptr<const Packet> pkt, CustomHeader ch)
{
	uint32_t src_id = ip_to_node_id(Ipv4Address(ch.sip));
	uint32_t dst_id = ip_to_node_id(Ipv4Address(ch.dip));
	uint32_t port_id = ch.udp.sport;
	uint32_t flow_id = flowMap[src_id][dst_id][port_id];
	uint32_t pkt_seq = ch.udp.seq / packet_payload_size;
	uint64_t pkt_delay = (Simulator::Now().GetNanoSeconds() - pktStartTime[flow_id][pkt_seq]) / 1000;
	pktDelay[flow_id][pkt_delay]++;
}

int main(int argc, char *argv[])
{
	clock_t begint, endt;
	begint = clock();
#ifndef PGO_TRAINING
	if (argc > 1)
#else
	if (true)
#endif
	{
		//Read the configuration file
		std::ifstream conf;
#ifndef PGO_TRAINING
		conf.open(argv[1]);
#else
		conf.open(PATH_TO_PGO_CONFIG);
#endif
		while (!conf.eof())
		{
			std::string key;
			conf >> key;
			if (key.compare("ENABLE_QCN") == 0){
				conf >> enable_qcn;
			}
			else if (key.compare("USE_DYNAMIC_PFC_THRESHOLD") == 0){
				conf >> use_dynamic_pfc_threshold;
			}
			else if (key.compare("CLAMP_TARGET_RATE") == 0){
				conf >> clamp_target_rate;
			}
			else if (key.compare("PAUSE_TIME") == 0){
				conf >> pause_time;
			}
			else if (key.compare("PACKET_PAYLOAD_SIZE") == 0){
				conf >> packet_payload_size;
			}
			else if (key.compare("L2_CHUNK_SIZE") == 0){
				conf >> l2_chunk_size;
			}
			else if (key.compare("L2_ACK_INTERVAL") == 0){
				conf >> l2_ack_interval;
			}
			else if (key.compare("L2_BACK_TO_ZERO") == 0){
				conf >> l2_back_to_zero;
			}
			else if (key.compare("TOPOLOGY_FILE") == 0){
				conf >> topology_file;
				topof.open(topology_file.c_str());
				topof >> node_num;
			}
			else if (key.compare("FLOW_FILE") == 0){
				conf >> flow_file;
			}
			else if (key.compare("SIMULATOR_STOP_TIME") == 0){
				conf >> simulator_stop_time;
			}
			else if (key.compare("ALPHA_RESUME_INTERVAL") == 0){
				for(uint32_t i = 0; i < node_num; i++){
					conf >> hConf[i].alpha_resume_interval;
				}
			}
			else if (key.compare("RP_TIMER") == 0){
				for(uint32_t i = 0; i < node_num; i++){
					conf >> hConf[i].rp_timer;
				}
			}
			else if (key.compare("EWMA_GAIN") == 0){	
				for(uint32_t i = 0; i < node_num; i++){
					conf >> hConf[i].ewma_shift;
					int v = 1 << int(hConf[i].ewma_shift);
					hConf[i].ewma_gain = 1.0 / v;
				}
			}
			else if (key.compare("FAST_RECOVERY_TIMES") == 0){
				conf >> fast_recovery_times;
			}
			else if (key.compare("RATE_AI") == 0){
				conf >> rate_ai;
			}
			else if (key.compare("RATE_HAI") == 0){
				conf >> rate_hai;
			}
			else if (key.compare("ERROR_RATE_PER_LINK") == 0){
				conf >> error_rate_per_link;
			}
			else if (key.compare("CC_MODE") == 0){
				conf >> cc_mode;
			}
			else if (key.compare("RATE_DECREASE_INTERVAL") == 0){
				for(uint32_t i = 0; i < node_num; i++){
					conf >> hConf[i].rate_decrease_interval;
				}
			}
			else if (key.compare("MIN_RATE") == 0){
				conf >> min_rate;
			}
			else if (key.compare("FCT_OUTPUT_FILE") == 0){
				conf >> fct_output_file;
			}
			else if (key.compare("HAS_WIN") == 0){
				conf >> has_win;
			}
			else if (key.compare("GLOBAL_T") == 0){
				conf >> global_t;
			}
			else if (key.compare("MI_THRESH") == 0){
				conf >> mi_thresh;
			}
			else if (key.compare("VAR_WIN") == 0){
				conf >> var_win;
			}
			else if (key.compare("FAST_REACT") == 0){
				conf >> fast_react;
			}
			else if (key.compare("U_TARGET") == 0){
				conf >> u_target;
			}
			else if (key.compare("INT_MULTI") == 0){
				conf >> int_multi;
			}
			else if (key.compare("RATE_BOUND") == 0){
				conf >> rate_bound;
			}
			else if (key.compare("ACK_HIGH_PRIO") == 0){
				conf >> ack_high_prio;
			}
			else if (key.compare("DCTCP_RATE_AI") == 0){
				conf >> dctcp_rate_ai;
			}
			else if (key.compare("PFC_OUTPUT_FILE") == 0){
				conf >> pfc_output_file;
			}
			else if (key.compare("LINK_DOWN") == 0){
				conf >> link_down_time >> link_down_A >> link_down_B;
			}
			else if (key.compare("BUFFER_SIZE") == 0){
				conf >> buffer_size;
			}
			else if (key.compare("MULTI_RATE") == 0){
				conf >> multi_rate;
			}
			else if (key.compare("SAMPLE_FEEDBACK") == 0){
				conf >> sample_feedback;
			}
			else if(key.compare("PINT_LOG_BASE") == 0){
				conf >> pint_log_base;
			}
			else if (key.compare("PINT_PROB") == 0){
				conf >> pint_prob;
			}
			else if (key.compare("PATH_OUTPUT_FILE") == 0){
				conf >> path_output_file;
			}
			else if (key.compare("CONFIG_OUTPUT_FILE") == 0){
				conf >> config_output_file;
			}
			// else if (key.compare("QLEN_MON_FILE") == 0){
			// 	conf >> qlen_mon_file;
			// }
			else if (key.compare("KMIN") == 0){
				for(uint32_t i = 0; i < node_num; i++){
					conf >> swConf[i].kmin;
				}
			}
			else if (key.compare("KMAX") == 0){
				for(uint32_t i = 0; i < node_num; i++){
					conf >> swConf[i].kmax;
				}
			}
			else if (key.compare("PMAX") == 0){
				for(uint32_t i = 0; i < node_num; i++){
					conf >> swConf[i].pmax;
				}
			}
            else if (key.compare("BUFFER_ALPHA") == 0){
				for(uint32_t i = 0; i < node_num; i++){
					conf >> swConf[i].alpha;
				}
			}
			else if (key.compare("LOG_PATH") == 0){
				conf >> logPath;
			};
			fflush(stdout);
		}
		conf.close();
	}
	else
	{
		std::cout << "Error: require a config file\n";
		fflush(stdout);
		return 1;
	}


	bool dynamicth = use_dynamic_pfc_threshold;

	Config::SetDefault("ns3::QbbNetDevice::PauseTime", UintegerValue(pause_time));
	Config::SetDefault("ns3::QbbNetDevice::QcnEnabled", BooleanValue(enable_qcn));
	Config::SetDefault("ns3::QbbNetDevice::DynamicThreshold", BooleanValue(dynamicth));

	// LogComponentEnable("QbbNetDevice", LOG_LEVEL_ALL);
	// set int_multi
	IntHop::multi = int_multi;
	// IntHeader::mode
	if (cc_mode == 7) // timely, use ts
		IntHeader::mode = IntHeader::TS;
	else if (cc_mode == 3) // hpcc, use int
		IntHeader::mode = IntHeader::NORMAL;
	else if (cc_mode == 10) // hpcc-pint
		IntHeader::mode = IntHeader::PINT;
	else // others, no extra header
		IntHeader::mode = IntHeader::NONE;

	// Set Pint
	if (cc_mode == 10){
		Pint::set_log_base(pint_log_base);
		IntHeader::pint_bytes = Pint::get_n_bytes();
		printf("PINT bits: %d bytes: %d\n", Pint::get_n_bits(), Pint::get_n_bytes());
	}

	//SeedManager::SetSeed(time(NULL));

	
	flowf.open(flow_file.c_str());
	uint32_t switch_num, link_num, trace_num;
	topof >> switch_num >> link_num;
	flowf >> flow_num;
	// std::cout << flow_num << std::endl;

	linkActivated = std::vector<bool> (link_num, false);

	//n.Create(node_num);
	std::vector<uint32_t> node_type(node_num, 0);
	for (uint32_t i = 0; i < switch_num; i++)
	{
		uint32_t sid;
		topof >> sid;
		node_type[sid] = 1;
	}
	for (uint32_t i = 0; i < node_num; i++){
		if (node_type[i] == 0)
			n.Add(CreateObject<Node>());
		else{
			Ptr<SwitchNode> sw = CreateObject<SwitchNode>();
			n.Add(sw);
			sw->SetAttribute("EcnEnabled", BooleanValue(enable_qcn));
		}
	}

	NS_LOG_INFO("Create nodes.");

	InternetStackHelper internet;
	internet.Install(n);

	//
	// Assign IP to each server
	//
	for (uint32_t i = 0; i < node_num; i++){
		if (n.Get(i)->GetNodeType() == 0){ // is server
			serverAddress.resize(i + 1);
			serverAddress[i] = node_id_to_ip(i);
		}
	}

	NS_LOG_INFO("Create channels.");

	//
	// Explicitly create the channels required by the topology.
	//

	Ptr<RateErrorModel> rem = CreateObject<RateErrorModel>();
	Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
	rem->SetRandomVariable(uv);
	uv->SetStream(50);
	rem->SetAttribute("ErrorRate", DoubleValue(error_rate_per_link));
	rem->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));

	FILE *pfc_file = fopen(pfc_output_file.c_str(), "w");
	
	QbbHelper qbb;
	Ipv4AddressHelper ipv4;
	for (uint32_t i = 0; i < link_num; i++)
	{
		uint32_t src, dst;
		std::string data_rate, link_delay;
		double error_rate;
		topof >> src >> dst >> data_rate >> link_delay >> error_rate;

		Ptr<Node> snode = n.Get(src), dnode = n.Get(dst);

		qbb.SetDeviceAttribute("DataRate", StringValue(data_rate));
		qbb.SetChannelAttribute("Delay", StringValue(link_delay));

		if (error_rate > 0)
		{
			Ptr<RateErrorModel> rem = CreateObject<RateErrorModel>();
			Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable>();
			rem->SetRandomVariable(uv);
			uv->SetStream(50);
			rem->SetAttribute("ErrorRate", DoubleValue(error_rate));
			rem->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
			qbb.SetDeviceAttribute("ReceiveErrorModel", PointerValue(rem));
		}
		else
		{
			qbb.SetDeviceAttribute("ReceiveErrorModel", PointerValue(rem));
		}

		fflush(stdout);

		// Assigne server IP
		// Note: this should be before the automatic assignment below (ipv4.Assign(d)),
		// because we want our IP to be the primary IP (first in the IP address list),
		// so that the global routing is based on our IP
		NetDeviceContainer d = qbb.Install(snode, dnode);
		if (snode->GetNodeType() == 0){
			Ptr<Ipv4> ipv4 = snode->GetObject<Ipv4>();
			ipv4->AddInterface(d.Get(0));
			ipv4->AddAddress(1, Ipv4InterfaceAddress(serverAddress[src], Ipv4Mask(0xff000000)));
		}
		if (dnode->GetNodeType() == 0){
			Ptr<Ipv4> ipv4 = dnode->GetObject<Ipv4>();
			ipv4->AddInterface(d.Get(1));
			ipv4->AddAddress(1, Ipv4InterfaceAddress(serverAddress[dst], Ipv4Mask(0xff000000)));
		}

		nodeLink[i].insert(src);
		nodeLink[i].insert(dst);
		// used to create a graph of the topology
		nbr2if[snode][dnode].idx = DynamicCast<QbbNetDevice>(d.Get(0))->GetIfIndex();
		nbr2if[snode][dnode].up = true;
		nbr2if[snode][dnode].delay = DynamicCast<QbbChannel>(DynamicCast<QbbNetDevice>(d.Get(0))->GetChannel())->GetDelay().GetTimeStep();
		nbr2if[snode][dnode].bw = DynamicCast<QbbNetDevice>(d.Get(0))->GetDataRate().GetBitRate();
		nbr2if[dnode][snode].idx = DynamicCast<QbbNetDevice>(d.Get(1))->GetIfIndex();
		nbr2if[dnode][snode].up = true;
		nbr2if[dnode][snode].delay = DynamicCast<QbbChannel>(DynamicCast<QbbNetDevice>(d.Get(1))->GetChannel())->GetDelay().GetTimeStep();
		nbr2if[dnode][snode].bw = DynamicCast<QbbNetDevice>(d.Get(1))->GetDataRate().GetBitRate();

		// This is just to set up the connectivity between nodes. The IP addresses are useless
		char ipstring[16];
		sprintf(ipstring, "10.%d.%d.0", i / 254 + 1, i % 254 + 1);
		ipv4.SetBase(ipstring, "255.255.255.0");
		ipv4.Assign(d);

		// setup PFC 
		DynamicCast<QbbNetDevice>(d.Get(0))->TraceConnectWithoutContext("QbbPfc", MakeBoundCallback (&get_pfc, pfc_file, DynamicCast<QbbNetDevice>(d.Get(0))));
		DynamicCast<QbbNetDevice>(d.Get(1))->TraceConnectWithoutContext("QbbPfc", MakeBoundCallback (&get_pfc, pfc_file, DynamicCast<QbbNetDevice>(d.Get(1))));

		// if(logPath)
		// {
		// 	DynamicCast<QbbNetDevice>(d.Get(0))->TraceConnectWithoutContext("Header", MakeBoundCallback (&qbb_enqueue, i, DynamicCast<QbbNetDevice>(d.Get(0))));
		// 	DynamicCast<QbbNetDevice>(d.Get(1))->TraceConnectWithoutContext("Header", MakeBoundCallback (&qbb_enqueue, i, DynamicCast<QbbNetDevice>(d.Get(1))));
		// }

		// DynamicCast<QbbNetDevice>(d.Get(0))->TraceConnectWithoutContext("HostSend", MakeBoundCallback (&host_send, i));
		// DynamicCast<QbbNetDevice>(d.Get(0))->TraceConnectWithoutContext("HostReceive", MakeBoundCallback (&host_receive, i));

	}
	nic_rate = get_nic_rate(n);

	// config switch
	for (uint32_t i = 0; i < node_num; i++){
		if (n.Get(i)->GetNodeType() == 1){ // is switch
			Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(n.Get(i));
			uint32_t shift = 3; // by default 1/8
			for (uint32_t j = 1; j < sw->GetNDevices(); j++){
				Ptr<QbbNetDevice> dev = DynamicCast<QbbNetDevice>(sw->GetDevice(j));
				// set ecn
				uint64_t rate = dev->GetDataRate().GetBitRate();
				uint32_t buffer_alpha = swConf[i].alpha;
				sw->m_mmu->ConfigEcn(j, swConf[i].kmin, swConf[i].kmax, swConf[i].pmax);


				// set pfc
				uint64_t delay = DynamicCast<QbbChannel>(dev->GetChannel())->GetDelay().GetTimeStep();
				uint32_t headroom = rate * delay / 8 / 1000000000 * 3;
				// std::cout << "rate=" << rate << ",delay=" << delay << ",headroom" << headroom << std::endl;
				sw->m_mmu->ConfigHdrm(j, headroom);

				// set pfc alpha, proportional to link bw
				sw->m_mmu->pfc_a_shift[j] = shift - buffer_alpha;
				while (rate > nic_rate && sw->m_mmu->pfc_a_shift[j] > 0){
					sw->m_mmu->pfc_a_shift[j]--;
					rate /= 2;
				}
			}
			sw->m_mmu->ConfigNPort(sw->GetNDevices()-1);
			sw->m_mmu->ConfigBufferSize(buffer_size* 1024 * 1024);
			sw->m_mmu->node_id = sw->GetId();
		}
	}

	#if ENABLE_QP
	FILE *fct_output = fopen(fct_output_file.c_str(), "w");
	// schedule buffer monitor
	// FILE *qlen_output = fopen(qlen_mon_file.c_str(), "w");
	// FILE *pfc_file = fopen(qlen_mon_file.c_str(), "w");

	// Simulator::Schedule(NanoSeconds(qlen_mon_start), &monitor_buffer, qlen_output, &n);
	
	// FILE *path_output = fopen(path_output_file.c_str(), "w");
	// FILE *config_output = fopen(config_output_file.c_str(), "w");
	fprintf(fct_output, "fid,size,src,dst,starttime,fct,standalone_fct\n");
	//
	// install RDMA driver
	//
	for (uint32_t i = 0; i < node_num; i++){
		if (n.Get(i)->GetNodeType() == 0){ // is server
			// create RdmaHw
			Ptr<RdmaHw> rdmaHw = CreateObject<RdmaHw>();
			rdmaHw->SetAttribute("ClampTargetRate", BooleanValue(clamp_target_rate));
			rdmaHw->SetAttribute("AlphaResumInterval", DoubleValue(hConf[i].alpha_resume_interval));
			rdmaHw->SetAttribute("RPTimer", DoubleValue(hConf[i].rp_timer));
			rdmaHw->SetAttribute("FastRecoveryTimes", UintegerValue(fast_recovery_times));
			rdmaHw->SetAttribute("EwmaGain", DoubleValue(hConf[i].ewma_gain));
			rdmaHw->SetAttribute("RateAI", DataRateValue(DataRate(rate_ai)));
			rdmaHw->SetAttribute("RateHAI", DataRateValue(DataRate(rate_hai)));
			rdmaHw->SetAttribute("L2BackToZero", BooleanValue(l2_back_to_zero));
			rdmaHw->SetAttribute("L2ChunkSize", UintegerValue(l2_chunk_size));
			rdmaHw->SetAttribute("L2AckInterval", UintegerValue(l2_ack_interval));
			rdmaHw->SetAttribute("CcMode", UintegerValue(cc_mode));
			rdmaHw->SetAttribute("RateDecreaseInterval", DoubleValue(hConf[i].rate_decrease_interval));
			rdmaHw->SetAttribute("MinRate", DataRateValue(DataRate(min_rate)));
			rdmaHw->SetAttribute("Mtu", UintegerValue(packet_payload_size));
			rdmaHw->SetAttribute("MiThresh", UintegerValue(mi_thresh));
			rdmaHw->SetAttribute("VarWin", BooleanValue(var_win));
			rdmaHw->SetAttribute("FastReact", BooleanValue(fast_react));
			rdmaHw->SetAttribute("MultiRate", BooleanValue(multi_rate));
			rdmaHw->SetAttribute("SampleFeedback", BooleanValue(sample_feedback));
			rdmaHw->SetAttribute("TargetUtil", DoubleValue(u_target));
			rdmaHw->SetAttribute("RateBound", BooleanValue(rate_bound));
			rdmaHw->SetAttribute("DctcpRateAI", DataRateValue(DataRate(dctcp_rate_ai)));
			rdmaHw->SetPintSmplThresh(pint_prob);
			// create and install RdmaDriver
			Ptr<RdmaDriver> rdma = CreateObject<RdmaDriver>();
			Ptr<Node> node = n.Get(i);
			rdma->SetNode(node);
			rdma->SetRdmaHw(rdmaHw);

			node->AggregateObject (rdma);
			rdma->Init();
			rdma->TraceConnectWithoutContext("QpComplete", MakeBoundCallback (qp_finish, fct_output));
		}
	}
	#endif

	// set ACK priority on hosts
	if (ack_high_prio)
		RdmaEgressQueue::ack_q_idx = 0;
	else
		RdmaEgressQueue::ack_q_idx = 3;

	// setup routing
	CalculateRoutes(n);
	SetRoutingEntries();

	//
	// get BDP and delay
	//
	maxRtt = maxBdp = 0;
	for (uint32_t i = 0; i < node_num; i++){
		if (n.Get(i)->GetNodeType() != 0)
			continue;
		for (uint32_t j = 0; j < node_num; j++){
			if (n.Get(j)->GetNodeType() != 0)
				continue;
			uint64_t delay = pairDelay[n.Get(i)][n.Get(j)];
			uint64_t txDelay = pairTxDelay[n.Get(i)][n.Get(j)];
			uint64_t rtt = delay * 2 + txDelay;
			uint64_t bw = pairBw[i][j];
			uint64_t bdp = rtt * bw / 1000000000/8; 
			pairBdp[n.Get(i)][n.Get(j)] = bdp;
			pairRtt[i][j] = rtt;
			if (bdp > maxBdp)
				maxBdp = bdp;
			if (rtt > maxRtt)
				maxRtt = rtt;
		}
	}
	// printf("maxRtt=%lu maxBdp=%lu\n", maxRtt, maxBdp);

	//
	// setup switch CC
	//
	for (uint32_t i = 0; i < node_num; i++){
		if (n.Get(i)->GetNodeType() == 1){ // switch
			Ptr<SwitchNode> sw = DynamicCast<SwitchNode>(n.Get(i));
			sw->SetAttribute("CcMode", UintegerValue(cc_mode));
			sw->SetAttribute("MaxRtt", UintegerValue(maxRtt));
		}
	}

	// dump link speed to trace file
	// {
	// 	SimSetting sim_setting;
	// 	for (auto i: nbr2if){
	// 		for (auto j : i.second){
	// 			uint16_t node = i.first->GetId();
	// 			uint8_t intf = j.second.idx;
	// 			uint64_t bps = DynamicCast<QbbNetDevice>(i.first->GetDevice(j.second.idx))->GetDataRate().GetBitRate();
	// 			sim_setting.port_speed[node][intf] = bps;
	// 		}
	// 	}
	// 	sim_setting.win = maxBdp;
	// 	sim_setting.Serialize(trace_output);
	// }

	Ipv4GlobalRoutingHelper::PopulateRoutingTables();

	NS_LOG_INFO("Create Applications.");

	Time interPacketInterval = Seconds(0.0000005 / 2);

	// maintain port number for each host
	for (uint32_t i = 0; i < node_num; i++){
		if (n.Get(i)->GetNodeType() == 0)
			for (uint32_t j = 0; j < node_num; j++){
				if (n.Get(j)->GetNodeType() == 0)
					portNumder[i][j] = 10000; // each host pair use port number from 10000
			}
	}

	flow_input.idx = 0;
	if (flow_num > 0){
		ReadFlowInput();
		Simulator::Schedule(Seconds(flow_input.start_time)-Simulator::Now(), ScheduleFlowInputs);
	}

	topof.close();

	// schedule link down
	if (link_down_time > 0){
		Simulator::Schedule(Seconds(2) + MicroSeconds(link_down_time), &TakeDownLink, n, n.Get(link_down_A), n.Get(link_down_B));
	}


	//
	// Now, do the actual simulation.
	//
	std::cout << "Running Simulation.\n";
	fflush(stdout);
	NS_LOG_INFO("Run Simulation.");
	Simulator::Stop(Seconds(5));
	Simulator::Run();
	Simulator::Destroy();
	NS_LOG_INFO("Done.");

	endt = clock();
	std::cout << "Simulation Completed in " << (double)(endt - begint) / CLOCKS_PER_SEC << " Seconds\n";

	fclose(fct_output);
	// fclose(qlen_output);
	// if(logPath)
	// {
	// 	for(uint i=0;i<flow_num;i++)
	// 	{
	// 		fprintf(path_output, "%u ", i);
	// 		// for(auto it:flowPath[i])
	// 		for(auto it:flowNode[i])
	// 		{
	// 			fprintf(path_output, "%u ", it);
	// 			linkActivated[it] = true;
	// 		}
	// 		fprintf(path_output, "\n");
	// 	}
	// }
	// fclose(path_output);

	// // uint32_t host_id = 16, tor_id = 24, agg_id = 32, core_id = 36;
	// uint32_t host_id = 32, tor_id = 40, agg_id = 44, core_id = 44;
	// if (node_num == 208)
	// {
	// 	host_id = 128;
	// 	tor_id = 160;
	// 	agg_id = 192;
	// 	core_id = 208;
	// }
	// else if (node_num == 1344)
	// {
	// 	host_id = 1024;
	// 	tor_id = 1024 + 128;
	// 	agg_id = 1024 + 128 + 128;
	// 	core_id = 1024 + 128 + 128 + 64;
	// }
	// fprintf(config_output, "node,type,kmin,kmax,pmax,buffer_alpha,t_alpha,t_inc,t_dec,g\n");
	// for(uint i = 0;i < node_num; i++)
	// {
	// 	if(node_type[i] == 0)
	// 	{
	// 		fprintf(config_output, "%u,host,0,0,0,0,%.0f,%.0f,%.0f,%u\n", i, hConf[i].alpha_resume_interval, hConf[i].rp_timer, hConf[i].rate_decrease_interval, hConf[i].ewma_shift);
	// 	}
	// 	else
	// 	{
	// 		if (i < tor_id)
	// 		{
	// 			fprintf(config_output, "%u,tor,%u,%lu,%.2f,%u,0,0,0,0\n", i, swConf[i].kmin, swConf[i].kmax, swConf[i].pmax, swConf[i].alpha);
	// 		}
	// 		else if (i < agg_id)
	// 		{
	// 			fprintf(config_output, "%u,agg,%u,%lu,%.2f,%u,0,0,0,0\n", i, swConf[i].kmin, swConf[i].kmax, swConf[i].pmax, swConf[i].alpha);
	// 		}
	// 		else if (i < core_id)
	// 		{
	// 			fprintf(config_output, "%u,core,%u,%lu,%.2f,%u,0,0,0,0\n", i, swConf[i].kmin, swConf[i].kmax, swConf[i].pmax, swConf[i].alpha);
	// 		}
	// 	}
	// }

	// std::unordered_map<uint32_t, map<uint64_t, uint32_t>>::iterator it;
	// for(it = pktDelay.begin(); it!=pktDelay.end(); it++)
	// {
	// 	map<uint64_t, uint32_t>::iterator it2;
	// 	for(it2 = it->second.begin(); it2!=it->second.end(); it2++)
	// 	{
	// 		fprintf(delay_output, "%u %lu %u\n", it->first, it2->first, it2->second);
	// 	}
	// }

}
