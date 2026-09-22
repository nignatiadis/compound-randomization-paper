# Validate directly against the published supplementary workbook (requires readxl).
p <- as.data.frame(readxl::read_excel('source/pmic70044-sup-0001-suppmat.xlsx',sheet='data'))
r <- read.csv('results/authors_model_reconstructed.csv',check.names=FALSE)
matched <- lapply(seq_len(nrow(p)),function(i) {
  j<-which(r$Gene==p$Gene[i] & r$firstAA==p$firstAA[i] & r$lastAA==p$lastAA[i] & abs(r$mass-p$mass[i])<1e-7)
  stopifnot(length(j)==1)
  data.frame(feature_id=r$feature_id[j],published_feature_id=p$featureName[i],Gene=p$Gene[i],
    abs_logFC_difference=abs(r$logFC[j]-p$logFC[i]),
    abs_p_difference=abs(r$P.Value[j]-p$P.Value[i]),
    abs_q_difference=abs(r$adj.P.Val[j]-p$adj.P.Val[i]))
})
m<-do.call(rbind,matched)
stopifnot(nrow(m)==839,all(as.matrix(m[,4:6])<1e-10))
write.csv(m,'data/published_feature_crosswalk.csv',row.names=FALSE)
writeLines(c(paste('Published rows matched:',nrow(m)),
  paste('Maximum absolute logFC difference:',max(m$abs_logFC_difference)),
  paste('Maximum absolute p-value difference:',max(m$abs_p_difference)),
  paste('Maximum absolute BH adjusted p-value difference:',max(m$abs_q_difference))),
  'results/published_validation.txt')
print(readLines('results/published_validation.txt'))
