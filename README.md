# Content-Based Movie Recommender from Scratch

A command-line movie recommendation engine built in Python. This project takes a dataset of movies and suggests similar films based on shared content features like genre, cast, director, and writers.

The core algorithms for text vectorization (**TF-IDF**) and similarity calculation (**Cosine Similarity**) were implemented entirely from scratch, without relying on common machine learning libraries like Scikit-learn.

![Screenshot of the recommender in action](<img width="918" height="316" alt="image" src="https://github.com/user-attachments/assets/25e85f0d-7af7-47ea-a7f0-181a8450a5b3" />
)


---

## Features

-   **Interactive Command-Line Interface:** Enter a movie title and get instant recommendations.
-   **Content-Based Logic:** Recommendations are based on a movie's "fingerprint," derived from its genre, tagline, creative team, and cast.
-   **Custom Algorithms:** Demonstrates a foundational understanding of recommendation systems by implementing TF-IDF and Cosine Similarity from scratch.
-   **Data-Driven:** Uses the "IMDB Top 250 Movies" dataset to provide relevant and interesting suggestions.

---

## How It Works

The recommendation engine follows a classic content-based filtering pipeline:

1.  **Data Loading & Preparation:** The program first loads the movie dataset using Pandas. It then engineers a "feature soup" for each movie—a single string containing all relevant descriptive text (genre, cast, director, etc.).

2.  **Text Processing:** Each movie's "soup" is **tokenized** (split into individual words). An **Inverted Index** is then built to efficiently map each word to the list of movies it appears in.

3.  **Vectorization (TF-IDF):** The program calculates the **Term Frequency-Inverse Document Frequency (TF-IDF)** score for every word in every movie. This creates a numerical vector for each film, where each number represents how uniquely important a word is to that film's identity.

4.  **Similarity Calculation:** When a user enters a movie title, the program calculates the **Cosine Similarity** between that movie's vector and every other movie's vector. This score (from 0 to 1) measures how "similar" the two movies are based on their content.

5.  **Ranking & Recommendation:** Finally, the program sorts all movies by their similarity score in descending order and returns the top 5 matches to the user.

---

## Technology Stack

-   **Python 3**
-   **Pandas:** For efficient data loading and manipulation from the CSV file.
-   **Standard Libraries:** `re` (for tokenization), `math`, `collections` (for algorithms).

---

## Setup and Installation

Follow these steps to get the project running on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/Your-Repo-Name.git
    cd Your-Repo-Name
    ```

2.  **Get the Dataset:**
    -   This project uses the `IMDB Top 250 Movies.csv` dataset.
    -   Download it (e.g., from Kaggle or another source) and place the `.csv` file in the root directory of this project.

3.  **Install Dependencies:**
    -   The only external dependency is Pandas.
    ```bash
    pip install pandas
    ```
    *(Alternatively, if a `requirements.txt` file is provided: `pip install -r requirements.txt`)*

---

## Usage

1.  Navigate to the project directory in your terminal.
2.  Run the main script:
    ```bash
    python movie_code.py
    ```
3.  The program will load the data and then prompt you to enter a movie title.
4.  Type a title from the dataset (e.g., "The Dark Knight" or "Pulp Fiction") and press Enter.
5.  The top 5 recommendations will be displayed.
6.  To stop the program, type `exit` and press Enter.
