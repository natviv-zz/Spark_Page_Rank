val pagerankGraph: Graph[Double, Double] = graph
  // Associate the degree with each vertex
  .outerJoinVertices(graph.outDegrees) {
    (vid, vdata, deg) => deg.getOrElse(0)
  }
  // Set the weight on the edges based on the degree
  .mapTriplets(e => 1.0 / e.srcAttr)
  // Set the vertex attributes to the initial pagerank values
  .mapVertices((id, attr) => 1.0)

def vertexProgram(id: VertexId, attr: Double, msgSum: Double): Double =
  resetProb + (1.0 - resetProb) * msgSum
def sendMessage(id: VertexId, edge: EdgeTriplet[Double, Double]): Iterator[(VertexId, Double)] =
  Iterator((edge.dstId, edge.srcAttr * edge.attr))
def messageCombiner(a: Double, b: Double): Double = a + b
val initialMessage = 0.0
// Execute Pregel for a fixed number of iterations.
Pregel(pagerankGraph, initialMessage, numIter)(
  vertexProgram, sendMessage, messageCombiner)

