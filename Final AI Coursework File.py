import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Download the dataset from Kaggle using kagglehub
path = kagglehub.dataset_download("prajitdatta/movielens-100k-dataset")

# Load the Dataset File u.data that provides the user date and reinteret the file to output the rating data in an organized way
#This is the rating movies data
ratings = pd.read_csv(r"C:\Users\david\.cache\kagglehub\datasets\prajitdatta\movielens-100k-dataset\versions\1\ml-100k\u.data",
    sep = '\t',
    header = None,
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    )


movie_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action',
                  'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',  
                 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

# This is the movie columns data
movies =  pd.read_csv(r"C:\Users\david\.cache\kagglehub\datasets\prajitdatta\movielens-100k-dataset\versions\1\ml-100k\u.item",
    sep = '|',
    header = None,
    encoding ='latin-1',
    names = movie_columns
)           

# Loading user data
user_data = pd.read_csv(r"C:\Users\david\.cache\kagglehub\datasets\prajitdatta\movielens-100k-dataset\versions\1\ml-100k\u.user",
    sep = '|',
    header = None,
    names = ['user_id', 'age', 'gender', 'occupation', 'zip_code'],
    encoding = 'latin-1'
)

print("Datasets loaded successfully.")

# Data Exploration and Visualization
ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')


# Preparing the Data for Modeling
genre_columns = [ 
    'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
    'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]


genre_features = movies[['movie_id'] + genre_columns].set_index('movie_id')

user_avg_rating = ratings.groupby('user_id')['rating'].mean()
user_rating_std = ratings.groupby('user_id')['rating'].std().fillna(0)
user_rating_counts = ratings.groupby('user_id').size()

user_profiles = pd.DataFrame({
    'user_id': user_avg_rating.index,
    'avg_rating': user_avg_rating.values,
    'rating_std': user_rating_std.values,
    'rating_count': user_rating_counts.values
})

# Normalize features to have mean=0, std=1 for fair comparison
scaler = StandardScaler()
user_profiles_norm = user_profiles.copy()
user_profiles_norm[['avg_rating', 'rating_std', 'rating_count']] = scaler.fit_transform(
    user_profiles[['avg_rating', 'rating_std', 'rating_count']]
)


# Split into training (80%) and testing (20%) sets
train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

# Rows = users, Columns = movies, Values = ratings, Missing = 0
train_matrix = train_ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
test_matrix = test_ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

class KNNCollaborativeFilter:
    """
    K-Nearest Neighbors Collaborative Filtering
    - Finds users similar to you
    - Recommends what they liked
    """
    def __init__(self, k=5):
        """k = number of neighbors to consider"""
        self.k = k
        self.user_similarity = None
        self.user_movie_matrix = None
    
    def fit(self, user_movie_matrix):
        """
        Train the model: Calculate similarity between all users
        Input: user_movie_matrix (rows=users, columns=movies, values=ratings)
        """
        # Store the matrix
        self.user_movie_matrix = user_movie_matrix.fillna(0)
        
        # Calculate similarity: how alike are users based on rating patterns
        # cosine_similarity returns matrix where [i,j] = similarity between user i and j
        similarity = cosine_similarity(self.user_movie_matrix)
        
        # Convert to DataFrame for easier indexing
        self.user_similarity = pd.DataFrame(
            similarity,
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.index
        )
        return self
    
    def predict(self, user_id, movie_id):
        """
        Predict rating for a user-movie pair
        Algorithm:
        1. Find k users most similar to target user
        2. Get their ratings for this movie
        3. Weighted average (weight = similarity)
        """
        # Check if user and movie exist
        if user_id not in self.user_similarity.index or movie_id not in self.user_movie_matrix.columns:
            return 3.0  # Default average rating
        
        # Get k nearest neighbors (excluding the user themselves)
        # sort_values(ascending=False) puts highest similarity first
        neighbors = self.user_similarity[user_id].drop(user_id).sort_values(ascending=False).head(self.k)
        
        # Get ratings from neighbors for this movie
        neighbor_ratings = self.user_movie_matrix.loc[neighbors.index, movie_id]
        
        # Only consider neighbors who actually rated this movie (rating > 0)
        rated = neighbor_ratings[neighbor_ratings > 0]
        
        # If no neighbors rated this movie, return default
        if len(rated) == 0:
            return 3.0
        
        # Calculate weighted average: neighbors with higher similarity have more weight
        return np.average(rated.values, weights=neighbors[rated.index].values)

# Train the KNN model
knn_model = KNNCollaborativeFilter(k=5)
knn_model.fit(train_matrix)

def build_pref_vector(user_id, user_matrix, genres, threshold=3.5):
    """
    Build user preference vector from their highly-rated movies
    
    How it works:
    1. Find movies user rated >= threshold (e.g., 3.5 stars)
    2. Get genre features of those movies
    3. Weighted average by rating
    4. Result: vector showing user's preference for each genre
    """
    # Return uniform if user not found
    if user_id not in user_matrix.index:
        return np.ones(genres.shape[1]) / genres.shape[1]
    
    user_ratings = user_matrix.loc[user_id]
    
    # Find highly-rated movies
    highly_rated = user_ratings[user_ratings >= threshold].index.tolist()
    
    # If no highly-rated, use all rated movies
    if len(highly_rated) == 0:
        highly_rated = user_ratings[user_ratings > 0].index.tolist()
    
    # If no ratings, return uniform preference
    if len(highly_rated) == 0:
        return np.ones(genres.shape[1]) / genres.shape[1]
    
    # Find movies that have genre data
    available = [m for m in highly_rated if m in genres.index]
    if len(available) == 0:
        return np.ones(genres.shape[1]) / genres.shape[1]
    
    # Get genre vectors and ratings for these movies
    genre_vecs = genres.loc[available]
    ratings_vals = user_ratings.loc[available]
    
    # Calculate weighted average: higher-rated movies have more influence
    pref = np.average(genre_vecs.values, axis=0, weights=ratings_vals.values)
    
    # Normalize to sum to 1 (interpretable as probability distribution)
    return pref / (np.sum(pref) + 1e-10)

# Build preference vectors for all users
user_prefs = {}
for user_id in train_matrix.index:
    user_prefs[user_id] = build_pref_vector(user_id, train_matrix, genre_features)

class ContentBasedFilter:
    """
    Content-Based Filtering
    - Builds user preference profile (genre preferences)
    - Recommends movies with similar genres to profile
    """
    def __init__(self, user_prefs, genres):
        self.user_prefs = user_prefs
        self.genres = genres
    
    def predict(self, user_id, movie_id):
        """
        Predict rating based on genre similarity
        
        How it works:
        1. Get user's genre preference vector
        2. Get movie's genre vector
        3. Calculate cosine similarity (how similar are they?)
        4. Convert similarity [-1, 1] to rating [1, 5]
        """
        # Return default if user or movie not found
        if user_id not in self.user_prefs or movie_id not in self.genres.index:
            return 3.0
        
        # Get vectors
        user_pref = self.user_prefs[user_id]
        movie_feat = self.genres.loc[movie_id].values
        
        # Cosine similarity: dot product / (magnitudes)
        dot = np.dot(user_pref, movie_feat)
        norm_u = np.linalg.norm(user_pref)
        norm_m = np.linalg.norm(movie_feat)
        
        # Avoid division by zero
        if norm_u == 0 or norm_m == 0:
            return 3.0
        
        # Similarity ranges from -1 (opposite) to 1 (identical)
        sim = dot / (norm_u * norm_m)
        
        # Convert to rating: similarity=-1 → rating=1, sim=0 → rating=3, sim=1 → rating=5
        predicted_rating = 1 + (sim + 1) * 2
        
        # Clamp to valid range [1, 5]
        return max(1.0, min(5.0, predicted_rating))

# Train content-based model
content_model = ContentBasedFilter(user_prefs, genre_features)


# Sample test data for efficiency
test_sample = test_matrix.sample(min(500, len(test_matrix)), random_state=42)

# Generate predictions
knn_preds, content_preds, actuals = [], [], []

for user_id in test_sample.index:
    for movie_id in test_sample.columns:
        actual = test_sample.loc[user_id, movie_id]
        if actual > 0:
            knn_preds.append(knn_model.predict(user_id, movie_id))
            content_preds.append(content_model.predict(user_id, movie_id))
            actuals.append(actual)

# Calculate evaluation metrics
knn_preds = np.array(knn_preds)
content_preds = np.array(content_preds)
actuals = np.array(actuals)

knn_mae = mean_absolute_error(actuals, knn_preds)
content_mae = mean_absolute_error(actuals, content_preds)
knn_rmse = np.sqrt(mean_squared_error(actuals, knn_preds))
content_rmse = np.sqrt(mean_squared_error(actuals, content_preds))

print("Evaluation Results:")
print(f"  KNN-CF:        MAE={knn_mae:.4f}, RMSE={knn_rmse:.4f}")
print(f"  Content-Based: MAE={content_mae:.4f}, RMSE={content_rmse:.4f}")
print(f"  Improvement:   {((content_mae - knn_mae) / content_mae * 100):.1f}% better MAE with KNN")

# Your evaluation metrics (from the code above)
knn_mae = mean_absolute_error(actuals, knn_preds)
content_mae = mean_absolute_error(actuals, content_preds)
knn_rmse = np.sqrt(mean_squared_error(actuals, knn_preds))
content_rmse = np.sqrt(mean_squared_error(actuals, content_preds))

# Prepare data for the grouped bar chart
algorithms = ['MAE', 'RMSE']
knn_values = [knn_mae, knn_rmse]
content_values = [content_mae, content_rmse]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# X-axis positions for bars
x = np.arange(len(algorithms))
width = 0.35  # Width of each bar

# Create bars
bars1 = ax.bar(x - width/2, knn_values, width, label='KNN', color='#1f77b4')
bars2 = ax.bar(x + width/2, content_values, width, label='Content', color='#ff0000')

# Customize the plot
ax.set_title('Algorithm Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('Error (Lower is Better)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(algorithms, fontsize=11)
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add subtle background
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("Graph saved as 'algorithm_comparison.png'")