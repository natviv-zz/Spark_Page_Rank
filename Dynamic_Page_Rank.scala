var PR = Array.fill(n)( 1.0 )
val oldPR = Array.fill(n)( 0.0 )
while( max(abs(PR - oldPr)) > tol ) {
  swap(oldPR, PR)
  for( i <- 0 until n if abs(PR[i] - oldPR[i]) > tol ) {
    PR[i] = alpha + (1 - \alpha) * inNbrs[i].map(j => oldPR[j] / outDeg[j]).sum
  }
}
