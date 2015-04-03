#include "Galois/Galois.h"
#include "Galois/Timer.h"
#include "Galois/Graph/LCGraph.h"
#include "Lonestar/BoilerPlate.h"
#include <algorithm>
#include <vector>
using namespace std;
static const char* const name = "PAGERANK";
static const char* const desc = "Compute PAGE rank";
static const char* const url = 0;

typedef struct Node {
double prev_val;
double now_val;
} Node;
namespace cll = llvm::cl;
static cll::opt<std::string> inputGraphFilename(cll::Positional,
cll::desc("<training graph input file>"), cll::Required);
static cll::opt<unsigned> T("T", cll::desc("maximum number of iterations"), cll::init(50));
static cll::opt<double> alpha("alpha", cll::desc("ratio"), cll::init(0.15));
using Graph = Galois::Graph::LC_CSR_Graph<Node, double>::with_out_of_line_lockable<true>::type;
using GNode = Graph::GraphNode;
typedef std::pair<int, double> mypair;
bool mycompare(const mypair& l, const mypair& r)
{
return l.second > r.second;
}

struct gogo {
Graph& g;
gogo(Graph &_g): g(_g){}
void operator()(GNode n, Galois::UserContext<GNode>& ctx){
double sum = 0.0;
for (auto edge_it : g.out_edges(n, Galois::NONE)) {
GNode variable_node = g.getEdgeDst(edge_it);
Node& data = g.getData(variable_node, Galois::NONE);
double weight = g.getEdgeData(edge_it, Galois::NONE);
sum += data.prev_val * weight;
}
Node& data = g.getData(n, Galois::NONE);
data.now_val = sum;
}
};

void multiply(Graph &g)
{
auto ln = Galois::loopname("PAGERANK");
auto wl = Galois::wl<Galois::WorkList::dChunkedFIFO<32>>();
Galois::for_each(g.begin(), g.end(), gogo(g), ln, wl);
/* Slower way... */
/*
Galois::do_all(g.begin(), g.end(), [&](GNode n) {
double sum = 0.0;
for (auto edge_it : g.out_edges(n)) {

GNode variable_node = g.getEdgeDst(edge_it);
Node& data = g.getData(variable_node);
double weight = g.getEdgeData(edge_it);
sum += data.prev_val * weight;
}
Node& data = g.getData(n);
data.now_val = sum;
});
*/
}



int main(int argc, char** argv) {
LonestarStart(argc, argv, name, desc, url);
Galois::StatManager statManager;
Graph g;
Galois::Graph::readGraph(g, inputGraphFilename);
int numnodes = g.size();
double one_over_n = 1.0/numnodes;
Galois::do_all(g.begin(), g.end(), [&](GNode n) {
Node& data = g.getData(n);
data.prev_val = one_over_n;
});
vector<int> degree(numnodes);
for ( int i=0 ; i<numnodes ; i++)
degree[i] = std::distance(g.edge_begin(i), g.edge_end(i));
Galois::do_all(g.begin(), g.end(), [&](GNode n) {
double sum = 0.0;
for (auto edge_it : g.out_edges(n)) {
GNode variable_node = g.getEdgeDst(edge_it);
double& weight = g.getEdgeData(edge_it);
weight = weight/degree[variable_node];
}
});
Galois::StatTimer PagerankTime("pagerank");
PagerankTime.start();
for ( int i=0 ; i<T ; i++) {
multiply(g);
Galois::do_all(g.begin(), g.end(), [&](GNode n) {
Node& data = g.getData(n);
data.prev_val = data.now_val*(1-alpha) + alpha*one_over_n;


});
}
PagerankTime.stop();
printf("Time: %lf secs\n", PagerankTime.get()/1e3);
vector<mypair> results(numnodes);
for ( int i=0 ; i<numnodes ; i++ ) {
results[i].first = i+1;
results[i].second = g.getData(i).prev_val;
}
sort(results.begin(), results.end(), mycompare);
printf("Top Nodes: ");
for ( int i=0 ; i<10 ; i++ )
printf("%d:%.8lf ", results[i].first, results[i].second);
printf("\n");
return 0;
}
