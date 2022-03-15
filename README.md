# bee-friendly Achieving 1st Place during the CropThon

## Background Information
This project was implemented during the CropThon/Ideathon/Hackathon on 13.11.2021-14.11.2021.
The CropThon was presented by TechLabs Aachen, Bayer CropScience and the digitalHub Aachen.

The CropThon's task was to identify and cluster existing pesticide product formulas / agriculture data using tools from the field of data science and subsequently pitch them as product or start-up ideas to Bayer CropScience. 

## About this Project
Streamlit is having some problems lately. So it is possible that the App is not loading at the moment.
This Application can be used over the following link: https://share.streamlit.io/valantano/bee_friendly/main/app.py

### Problem-Statement: 
Not all pesticides are already tested for whether they harm bees or not. In our Dataset, we have 631 pesticides that are already tested for their honeybees_contact_kill_risk and 1686 pesticides that are not yet tested.

### Solution:
Instead of testing each pesticide we analyze the structure of the pesticide molecules and calculate the distance between the molecular structures using the inchi fingerprint. After that, we use UMAP for dimension reduction and apply a KNN-Classifier to the resulting 3D Plot.

### Performance:
The KNN-Classifier performs best for k=2: classifies 0.6878% of the already tested pesticides correctly. This is quite awesome if you consider that changing only a small part of a molecule can lead to completely different behavior.

### Challenges:
Sometimes the nearest already tested neighbor is very far away. Then the classifier should reject the classification because the molecular structure is not similar enough to make assumptions about similar behavior.

## Note
The Classifier is not a real KNN-Classifier because it also takes into account the distance of each Neighbor. So nearest neighbor is weighted higher than the neighbor which has the highest distance to the pesticide which shall be classified. This way the performance could be improved by nearly 1%

## Project Group
The Project was implemented by the following team members:
- Philipus Benizi Putra
- Jonathan Krahl
- Tom Stein
- Valentino Geuenich

https://www.linkedin.com/posts/tom1337stein_two-weeks-ago-i-went-to-aachen-together-activity-6870333676866048000-CXAW
