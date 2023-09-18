# Arabic-News-Clustering-for-Factuality-Estimation
Arabic News Clustering Full Code and files

This repository contains all the code that was used to run the interface of the Arabic News Clustering for Factuality Estimation project at the QCRI 2023 Summer Internship. We use specific models that were made to allow certain processes 
to occur such as the Check Worthiness on the main site. We also use FAISS to acquire a similarity score for the articles when the user searches. This is done using Cosine Similarity. Additionally, we use arabert, a pre-trained model 
on Arabic text since most of the articles used are written in Arabic. We also created embeddings for our articles that we saved in a NumPy file. Flask is then used to create the application that connects the Python code to the HTML pages 
that were created such as the QuerySite.html file. Within the Flask section of the Zaman_Website.py, there are also some functions that run when a specific site is entered or a specific button is clicked. 
