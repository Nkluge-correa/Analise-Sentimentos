# ----------------------------------------------------------------------------------------#
# Bibliotecas
# ----------------------------------------------------------------------------------------#
import pandas as pd
from google_play_scraper import Sort, reviews, app
from tqdm import tqdm

# ----------------------------------------------------------------------------------------#
# Scrape apps
# ----------------------------------------------------------------------------------------#
apps_ids = [
    'br.com.brainweb.ifood', 
    'com.cerveceriamodelo.modelonow',
    'com.mcdo.mcdonalds', 
    'habibs.alphacode.com.br',
    'com.xiaojukeji.didi.brazil.customer',
    'com.ubercab.eats', 
    'com.grability.rappi',
    'burgerking.com.br.appandroid', 
    'com.instagram.android',
    'com.tinder',
    'com.facebook.katana',
    'com.google.android.youtube',
    'com.zhiliaoapp.musically',
    'com.ubercab',
    'com.twitter.android',
    'org.telegram.messenger',
]

# ----------------------------------------------------------------------------------------#
# Raw Data
# ----------------------------------------------------------------------------------------#

app_infos = []

for ap in tqdm(apps_ids):
    info = app(ap, lang='en', country='us')
    del info['comments']
    app_infos.append(info)

app_reviews = []

for ap in tqdm(apps_ids):
    for score in list(range(1, 6)):
        for sort_order in [Sort.MOST_RELEVANT]:
            rvs, _ = reviews(
                ap,
                lang='pt',
                country='br',
                sort=sort_order,
                count= 0 if score == 3 else 1000,
                filter_score_with=score
            )
            for r in rvs:
                r['sortOrder'] = 'most_relevant' 
                r['appId'] = ap
            app_reviews.extend(rvs)

data = pd.DataFrame(app_reviews)
data.to_excel('data.xlsx', index=None, header=True)

# ----------------------------------------------------------------------------------------#
# Clean Data
# ----------------------------------------------------------------------------------------#

data = data.drop(['reviewId', 'userName', 'userImage',
                'thumbsUpCount', 'reviewCreatedVersion', 'at',
                'replyContent', 'repliedAt', 'sortOrder', 'appId'], axis=1)

def to_target(rating):
    rating = int(rating)
    if rating <= 2:
        return 0
    else: 
        return 1

data['score'] = data.score.apply(to_target)


l = list(data['content'])
new_l = []
for review in l:
    new_review = review.translate(str.maketrans('', '', string.punctuation))
    new_review = new_review.lower()
    new_review = unidecode.unidecode(new_review)
    new_l.append(new_review)

labelos = data['score']
labels = labelos.to_numpy()

df = pd.DataFrame()
df['content'] = new_l
df['score'] = labels
df.to_excel('data_clean.xlsx', index=None, header=True)