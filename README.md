# Detecting-Retinal-Cell-Classes-and-Ganglion-Cell-Subtypes
Detecting Retinal Cell Classes and Ganglion Cell Subtypes  Based on Transcriptome Data with Deep Transfer Learning

Purpose:  To  develop  and  assess  the  accuracy  of  deep  learning  models  that  identifies different  retinal  cell  types,  as  well  as  different  retinal  ganglion  cell  (RGC)  subtypes, based on patterns of single-cell RNA sequencing (scRNA-seq) in multiple data sets.

![image](https://user-images.githubusercontent.com/35179314/168642612-415a6d73-9619-4e85-b4c6-1713ae933a12.png)

# Data preparation
We used the Seurat to normalize and scale the count transcriptome data. We exclude cells with fewer than 200 expressed genes and removed genes that were expressed in fewer than 3 cells (based on Seurat). We then selected highly variable genes. More specifically, we computed dispersion (variance/mean) and the mean expression of each gene. Specifically, we put the dispersion measure for each gene across all cells into several bins based on their average expression. Within each bin, the z-normalized of the dispersion measure of all genes was calculated to identify highly variable genes. Based on Macosko et al., the z-score cutoff was 1.7, which is the default parameter in Seurat. We selected a z-score of 1.7 which selected about 2000 highly variable genes.

We prepared an example of processed datasets as follows: Merged_Expression.csv [gene, cell] is the merged source and target expression matrix. Merged_Labels.csv is the merged source and target cluster labels. Labels_of_Domains.csv is the merged source and target batch/domain labels. Labels_dict.csv is the 1-1 mapping among digital labels and cluster labels.


# Using Python script
An example of how to utilize script for classification and batch correction:

python main.py 

              --dataset_path path/to/input/files

              --result_path path/for/output/files

              --source_name batch name

              --target_name batch name

              --gpu_id GPU id to run
                           
The dataset_path must contain the four preprocessed CSV files as in dataset/processed_data folder. After running the model, there will be three output files in result_path: final_model_*.ckpt which has the trained model parameters (for example: weights and biases) and it can be loaded for label prediction. pred_labels_*.csv which contains the predicted cell label and corresponding confidence score (softmax probability). embeddings_*.csv which contains the batch-corrected low-dimensional embeddings (default is 256) for visualization.               
