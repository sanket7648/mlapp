from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llm_recommendations import llm_based_recommendations

app = Flask(__name__)

# Add session configuration
app.secret_key = "alskdjfwoeieiurlskdjfslkdjf"

# load files===========================================================================================================
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")

# database configuration---------------------------------------
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:hello32U!@localhost/ecom"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# Define your model class for the 'signup' table
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Define your model class for the 'signin' table
class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)


# Recommendations functions============================================================================================
# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text


def content_based_recommendations(train_data, item_name, top_n=10):
    # Check if the item name exists in the training data
    if item_name not in train_data['Name'].values:
        print(f"Item '{item_name}' not found in the training data.")
        return pd.DataFrame()

    print(f"Item '{item_name}' found. Generating recommendations...")

    # Create a TF-IDF vectorizer for item descriptions
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Apply TF-IDF vectorization to item descriptions
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])

    # Calculate cosine similarity between items based on descriptions
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    # Find the index of the item
    item_index = train_data[train_data['Name'] == item_name].index[0]

    # Get the cosine similarity scores for the item
    similar_items = list(enumerate(cosine_similarities_content[item_index]))

    # Sort similar items by similarity score in descending order
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    # Get the top N most similar items (excluding the item itself)
    top_similar_items = similar_items[1:top_n+1]

    # Get the indices of the top similar items
    recommended_item_indices = [x[0] for x in top_similar_items]

    # Get the details of the top similar items
    recommended_items_details = train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

    print(f"Recommended items: {recommended_items_details}")
    return recommended_items_details

# routes===============================================================================
# List of predefined image URLs
random_image_urls = [
    "static/img/img_1.png",
    "static/img/img_2.png",
    "static/img/img_3.png",
    "static/img/img_4.png",
    "static/img/img_5.png",
    "static/img/img_6.png",
    "static/img/img_7.png",
    "static/img/img_8.png",
]



@app.route("/")
def index():
    # Create a list of random image URLs for each product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html',trending_products=trending_products.head(8),truncate = truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price = random.choice(price))

@app.route("/main")
def main():
    # Check if the user is logged in
    if 'username' not in session:
        return redirect(url_for('signin'))

    # Initialize content_based_rec as an empty list if no recommendations are available
    content_based_rec = []

    # Pass the empty list to the template
    return render_template('main.html', content_based_rec=content_based_rec)

# routes
@app.route("/index")
def indexredirect():
    # Create a list of random image URLs for each product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    message = request.args.get('message', None)
    return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price=random.choice(price),
                           message=message)

@app.route("/signup", methods=['POST'])
def signup():
    username = request.form.get('signupUsername')
    email = request.form.get('signupEmail')
    password = request.form.get('signupPassword')

    # Validate that all fields are filled
    if not username or not email or not password:
        return "Please fill in all the required fields.", 400  # Return a 400 Bad Request response

    # Check if the username or email already exists in the database
    existing_user = Signup.query.filter((Signup.username == username) | (Signup.email == email)).first()
    if existing_user:
        return redirect(url_for('indexredirect', message="Username or email already exists."))

    # Create a new user and add to the database
    new_user = Signup(username=username, email=email, password=password)
    db.session.add(new_user)
    db.session.commit()

    # Redirect to the index page with a success message
    return redirect(url_for('indexredirect', message="User signed up successfully!"))

# Route for signup page
@app.route('/signin', methods=['POST'])
def signin():
    username = request.form.get('signinUsername')
    password = request.form.get('signinPassword')

    # Validate that all fields are filled
    if not username or not password:
        return "Please fill in all the required fields.", 400  # Return a 400 Bad Request response

    # Check if the user exists in the database
    user = Signin.query.filter_by(username=username, password=password).first()
    if user:
        session['username'] = username  # Store username in session
        return redirect(url_for('indexredirect', message="User signed in successfully!"))
    else:
        return redirect(url_for('indexredirect', message="Invalid username or password!"))

@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod')  # Get the product name from the form
        nbr = request.form.get('nbr')   # Get the number of recommendations from the form

        # Handle empty 'nbr' field
        if not nbr or not nbr.isdigit():
            nbr = 10  # Default value if 'nbr' is empty or invalid
        else:
            nbr = int(nbr)

        print(f"Product searched: {prod}, Number of recommendations: {nbr}")

        # Step 1: Search for the product in the dataset
        if prod in train_data['Name'].values:
            print(f"Product '{prod}' found in the dataset. Using content-based recommendations.")
            content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)
        else:
            print(f"Product '{prod}' not found in the dataset. Using LLM-based recommendations.")
            content_based_rec = llm_based_recommendations(train_data, prod, top_n=nbr)

        # Step 2: Check if recommendations are available
        if content_based_rec.empty:
            print("No recommendations found.")
            message = "No recommendations available for this product."
            return render_template('main.html', message=message, content_based_rec=[])
        else:
            print("Recommendations found:")
            print(content_based_rec)

            # Convert DataFrame to a list of dictionaries
            recommendations_list = content_based_rec.to_dict(orient='records')

            # Create a list of random image URLs for each recommended product
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(recommendations_list))]
            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]

            # Render the recommendations on the main.html page
            return render_template(
                'main.html',
                content_based_rec=recommendations_list,
                truncate=truncate,
                random_product_image_urls=random_product_image_urls,
                random_price=random.choice(price)
            )

if __name__=='__main__':
    app.run(debug=True)