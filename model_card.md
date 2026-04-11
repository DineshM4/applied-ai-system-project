# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  
Example: **MusiTune 1.0**  

---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 

Prompts:  

- What kind of recommendations does it generate  
- What assumptions does it make about the user  
- Is this for real users or classroom exploration  

This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users. It assumes user knows their prefferences that they had an idea of what they want before trying this reccomender. 

---

## 3. How the Model Works  

Explain your scoring approach in simple language.  

Prompts:  

- What features of each song are used (genre, energy, mood, etc.)  
- What user preferences are considered  
- How does the model turn those into a score  
- What changes did you make from the starter logic  

Avoid code here. Pretend you are explaining the idea to a friend who does not program.

The model takes in a table containing song data. Then after processing this data, the reccomender essentailly scores each song to the user's preferences in genre, mood, energy, valence, and acousticness. the closer something is to the user's preference the greater the score. Eventually this scored songs is sent a ranker which ranks by the top score and sends top however many sons user wants. I just changed fav_genre to a list of strings so that user can put multiple favorites and introduced a target valence for better recs.
---

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
- What genres or moods are represented  
- Did you add or remove data  
- Are there parts of musical taste missing in the dataset  

There are 20 songs in the catalog.
There is a diverse set of mood and genre from hip hop, r&b, pop rock lofi, etc. Moods include peaceful, euphoric, laid-back, intense, meloncholic, etc. 
---

## 5. Strengths  

Where does your system seem to work well  

Prompts:  

- User types for which it gives reasonable results  
- Any patterns you think your scoring captures correctly  
- Cases where the recommendations matched your intuition  

The more information the user gives for their preferences, the better the reccomender system works. Bonus points if the user's mood directly matches the moods of songs in the reccomender.
---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

Prompts:  

- Features it does not consider  
- Genres or moods that are underrepresented  
- Cases where the system overfits to one preference  
- Ways the scoring might unintentionally favor some users  


The biggest weakness I discovered during testing is the all or nothing style of mood scoring. A song either matched the user's preferred mood exactly and recieved full credit, or it didnt get any points at all. I didn't implement a partial credit system for moods so moods that feel similar, like "relaxed" and "chill" would only get 0.0 or 1.0. This makes it harder because of the amount of songs that share user's exact mood, because certain moods are more prominent than other moods. My recommender in a sense quietly gives up on an entire dimension of a user's taste rather than trying to approximate it because of this binary aspect.
---

## 7. Evaluation  

How you checked whether the recommender behaved as expected. 

Prompts:  

- Which user profiles you tested  
- What you looked for in the recommendations  
- What surprised you  
- Any simple tests or comparisons you ran  

No need for numeric metrics unless you created some.

I checked mainly 2 sets of user profiles, normal examples ones and contradictory profiles. I looked for recommendations mainly based on my mood and genre, with an added touch of accoustincness, and based on that top 5 recs the reccomender produced, the new weights seem to work a lot better than before to provide a more accurate reccomendation. However what surprised me was the implementation of binary scoring in some of my checks like mood (0.0 or 1.0). By making it an absolute yes or no, the reccomendation system itself has a hard time to be more accurate as its in a way wasting points for anything that doesnt directly match. I now understand the importance of vectorizing all data as it creates the closest gradient match in float representation rather than binary. All my tests came from weigth comprisement of the contradictory user profiles. 


---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
- Better ways to explain recommendations  
- Improving diversity among the top results  
- Handling more complex user tastes  

Have a more complex system that isn't hardcoded all the scoring logic or binary but rather uses vector logic and "real math" to score better. 
---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
- Something unexpected or interesting you discovered  
- How this changed the way you think about music recommendation apps  

I learned the importance of reccomender systems using vectors and gradients to highlight similarity. This process shifts the scoring from having to hardcoding certain aspects to having a more continuous accurate comparision system. Something interesting I discovered is how much influence the accousticness can to the reccomender. It may have a low weigth but the scale if defines from electronic to accoustic helps the reccomender give that slight edge to on song over another. Now I understand just how complex and accurate music reccomendation apps complete to give the user a reccomended playlist. I was also surpised by how even though my reccomender was very simple, it still did a decent job at giving an acceptable reccomendation(maybe because there were only so many songs), becuase the hardcoded fields carried most of the work. AI tools also helped pinpoint and infact came with some the contradictory_user_profiles which really helped change some logic to make a more accurate reccomender. Next time in this project when I'm ready, I want to try and look at real vector database styled implementations to change the project from a simple reccomender to a more complex one.