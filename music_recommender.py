from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

data = pd.read_csv('amazon_music_metadata.csv')
music_data = data.drop(['title', 'asin'], axis=1)
cosine_sim = cosine_similarity(music_data, music_data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    music_title = request.form['music_title']
    recommendations = get_recommendations(music_title)
    return render_template('recommendations.html', music_title=music_title, recommendations=recommendations)

def get_recommendations(title, cosine_sim=cosine_sim, data=data, top=10):
    recs=[]
    try:
        idx=-1
        for i,row in data.iterrows():
            if(row['title'].lower()==title.lower()):
                idx=i
                break
        if idx!=-1:
            sim_music = list(enumerate(cosine_sim[idx]))
            sim_music = sorted(sim_music, key=lambda x: x[1], reverse=True)

            m = min(len(sim_music), top+1)
            for i in range(1, m):
                recs.append(sim_music[i][0])

    except:
        recs=[]
    
    return list(data['title'].iloc[recs])

if __name__ == '__main__':
    app.run(debug=True)
