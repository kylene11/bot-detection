import json
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

# === Step 1: Load label TSV ===
labels_df = pd.read_csv("../data/uncleaned_datasets/cresci-rtbust-2019.tsv", sep="\t", names=["user_id", "label"])
labels_dict = dict(zip(labels_df["user_id"].astype(str), labels_df["label"]))

# === Step 2: Load JSON tweet data ===
with open("../data/uncleaned_datasets/cresci-rtbust-2019_tweets_prettier.json", "r") as f:
    data = json.load(f)

# === Step 3: Extract and label unique users ===
user_data = {}


for entry in data:
    user = entry.get("user", {})
    user_id = str(user.get("id"))
    
    if user_id not in user_data and user_id in labels_dict:
        created_at = datetime.strptime(user.get("created_at"), "%a %b %d %H:%M:%S %z %Y")
        reference_time = datetime(2019, 5, 15, 16, 0, 19, tzinfo=created_at.tzinfo)  # now tz-aware
        account_age_days = max((reference_time - created_at).days, 1)


        user_data[user_id] = {
            "user_id": user_id,
            "name_length": len(user.get("name", "")),
            #unique twitter handle
            "screen_name_length": len(user.get("screen_name", 0)),
            "description_length": len(user.get("description", 0)),
           
            
         
            "protected": int(user.get("protected", False)),
            "verified": int(user.get("verified", False)),

            "followers_count": user.get("followers_count", 0),
            "friends_count": user.get("friends_count", 0),
            #no of tweets
            "statuses_count": user.get("statuses_count", 0),
            #no of likes
            "favourites_count": user.get("favourites_count", 0),
           
        
            "default_profile_image": int(user.get("default_profile_image", False)),
            "account_age_days": account_age_days,
            "label": 1 if labels_dict[user_id] == "bot" else 0
        }

# === Step 4: Convert to DataFrame and add engineered features ===
df = pd.DataFrame(user_data.values())
df['following_vs_follower_ratio'] = df['friends_count'] / (df['followers_count'] + 1) #avoid 0 division 
df['tweet_rate'] = df['statuses_count'] / df['account_age_days']
df["followers_growth_rate"] = df["followers_count"] / df["account_age_days"]
df["friends_growth_rate"] = df["friends_count"] / df["account_age_days"]
df["favourites_growth_rate"] = df["favourites_count"] / df["account_age_days"]

##drop highly skewed cols + noise
# for col in ['protected', 'verified', 'default_profile_image']:
#     print(df[col].value_counts(normalize=True))

dropped_cols = ['protected', 'verified', 'default_profile_image',"followers_count", "friends_count", "statuses_count", "favourites_count"]
df.drop(columns=dropped_cols, inplace=True)


# === Step 5: Train/validation split ===
X = df.drop(columns=["user_id", "label"])
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Step 6: Save splits === (try to use parquet for faster reading, btr for larger datasets)
X_train.to_parquet("../data/X_train.parquet", index=False)
X_val.to_parquet("../data/X_val.parquet", index=False)
y_train.to_frame().to_parquet("../data/y_train.parquet", index=False)
y_val.to_frame().to_parquet("../data/y_val.parquet", index=False)


print("✅ Dataset prepared and splits saved.")
