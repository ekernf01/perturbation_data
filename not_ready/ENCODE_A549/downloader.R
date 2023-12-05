setwd("~/Desktop/jhu/research/projects/perturbation_prediction/cell_type_knowledge_transfer/perturbation_data/not_ready/ENCODE_A549/")
library(biomaRt)
mart <- useMart(dataset="hsapiens_gene_ensembl",biomart='ensembl')

#' Merge duplicates and retain ncbi ID if no HGNC id available
#'
handle_mismatched_ids = function(X, desired_ids) {
  missing_ids = setdiff(desired_ids, X$entrezgene_id)
  X %>%
    rbind(data.frame(entrezgene_id = missing_ids, hgnc_symbol = "")) %>%
    dplyr::group_by( entrezgene_id) %>% 
    dplyr::summarise(hgnc_symbol = paste0(hgnc_symbol, collapse = ";")) %>%
    dplyr::mutate(   hgnc_symbol = ifelse(hgnc_symbol=="", paste0("ENTREZ_", entrezgene_id), hgnc_symbol))
}

files = read.table(
  header = T, 
  text=
"
accession contents
GSE144662 post_dex_removal
GSE144660 overexpression
GSE91305 dex-0h
GSE91198 dex-0.5h
GSE91311 dex-1h
GSE91358 dex-2h
GSE91303 dex-3h
GSE91243 dex-4h
GSE91281 dex-5h
GSE91229 dex-6h
GSE91255 dex-7h
GSE91284 dex-8h
GSE91222 dex-10h
GSE91212 dex-12h
"
)
for(i in 1:nrow(files)){
  urld <- "https://www.ncbi.nlm.nih.gov/geo/download/?format=file&type=rnaseq_counts"
  path <- paste0(urld, 
                 "&acc=", 
                 files[i, "accession"], 
                 "&file=", 
                 files[i, "accession"], 
                 "_raw_counts_GRCh38.p13_NCBI.tsv.gz");
  tbl <- as.matrix(data.table::fread(path, header=T, colClasses="integer"), rownames=1)
  hgnc_symbols = biomaRt::select(mart, keys=rownames(tbl), columns=c('hgnc_symbol', 'entrezgene_id'), keytype='entrezgene_id')
  hgnc_symbols %<>% handle_mismatched_ids(desired_ids = rownames(tbl))
  tbl %>% rowSums %>% sort %>% tail
  rownames(tbl)  = setNames(hgnc_symbols$hgnc_symbol, hgnc_symbols$entrezgene_id)[as.character(rownames(tbl))]
  tbl %>% rowSums %>% sort %>% tail
  write.csv(tbl, paste0(files[i, "accession"], "_", files[i, "contents"], ".tsv"))
}
