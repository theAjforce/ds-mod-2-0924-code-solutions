# Recommendation System
This notebook contains my first attempt at a recommendation system using a combination of "item to item" with "user to user" correlation.

## Project Overview

### Goal
Create a recommendation system that provides the best poetry book recommendations for users.

### Techniques Used
In this project I used a hybrid approach combining collaborative and content-based methods for a more robust recommendation system. I achieved this by creating a linear combination for each user containing their ratings of their previously read books and then multiplying it across the utility matrix and then taking the sum of each book to find which ones had the highest correlation.

### Metrics
So obviously there is no preexisting test set to check for accuracy, and admittedly the book names themselves aren't exactly readable (they are all integers). So I instead created multiple KNearestNeighbors models and then passed each user through it simultaneously to see which model had the most TRUE predictions. A TRUE prediction was noted as a recommendation to the user that they had already previously rated 3 or higher.

SVD reduction matches with various K-values:    | Non SVD-Reduced matches: 12553
k=1     6451                                    | 
k=2     9886                                    |
k=3    10067                                    |
k=4    10299                                    |



### Final Model
I ended up finding that after trying multiple variations of SVD reduction that none of them gave the same amount of matches as the non SVD-reduced model I had started with.

### Data Source
<https://cseweb.ucsd.edu/~jmcauley/datasets.html#goodreads>

### Primary Libraries
The libraries I used primarily were Pandas and Sklearn.

## Usage
If you want to test the recommendation system yourself first:
1. Clone the repository: git clone https://github.com/theAjforce/ds-mod-2-0924-code-solutions/tree/recommendation-systems-project/recommendation-systems-project.git
2. Run the file inference_pipeline.py
3. Enter the UserID, example IDs: "0004ae25e3cf5f5a44b6f1ccfdd3d343","15147f5dde107b2c1fe888f88b28c61b","3ca7375dba942a760e53b726c472a7dd","660315d58b3755548cd044eb688914f2"
4. Examine the results!

