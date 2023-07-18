from flask import Flask, render_template, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from newspaper import Article
import random
import csv
from bs4 import BeautifulSoup
import requests
import os
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('model.pkl', 'rb'))
dataframe = pd.read_csv('news.csv')
x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


def shuffle_csv_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        data = list(csv.reader(csvfile))
        header = data[0]
        rows = data[1:]
        random.shuffle(rows)

    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)


def scrape_and_save_news():
    # Scrape real news from CNN
    cnn_url = 'https://edition.cnn.com/politics'
    cnn_response = requests.get(cnn_url)
    cnn_soup = BeautifulSoup(cnn_response.content, 'html.parser')
    cnn_div = cnn_soup.find('div', class_='container__field-links container_list-headlines__field-links')
    cnn_anchors = cnn_div.find_all('a', class_='container__link container_list-headlines__link')
    news_map = {}
    for anchor in cnn_anchors:
        news_url = "https://edition.cnn.com/" + anchor.get('href')
        news_response = requests.get(news_url)
        news_soup = BeautifulSoup(news_response.content, 'html.parser')
        news_title = news_soup.find(class_='headline__text inline-placeholder')
        if news_title:
            news_title = news_title.text.strip()
            div = news_soup.find(class_='article__content')
            paragraphs = div.find_all('p')
            news_content = ''
            for p in paragraphs:
                if 'footnote' in p.get('class', []):
                    continue
                text = p.text.strip()
                news_content += text + ' '
            news_map[news_title] = news_content

    # Scrape fake news from The Onion
    onion_url = 'https://www.theonion.com/politics/news-in-brief'
    onion_response = requests.get(onion_url)
    onion_soup = BeautifulSoup(onion_response.content, 'html.parser')
    onion_divs = onion_soup.find_all("div", class_="sc-cw4lnv-5 dYIPCV")
    fake_news_map = {}
    for div in onion_divs:
        anchor_tags = div.find_all("a")
        for anchor in anchor_tags:
            href = anchor.get("href")
            link_response = requests.get(href)
            link_soup = BeautifulSoup(link_response.text, "html.parser")
            fake_news_title = link_soup.find(class_='sc-1efpnfq-0 dAlcTj')
            if fake_news_title:
                fake_news_title = fake_news_title.text.strip()
                target_div = link_soup.find("div", class_="sc-r43lxo-1 cwnrYD")
                if target_div:
                    p_tag = target_div.find("p", class_="sc-77igqf-0 fnnahv")
                    if p_tag:
                        text = p_tag.get_text(strip=True)
                        fake_news_map[fake_news_title] = text

    # Read existing titles from the CSV file
    existing_titles = set()
    try:
        with open('news_data.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                existing_titles.add(row[1])
    except FileNotFoundError:
        pass

    # Append only new news to the CSV file
    with open('news_data.csv', 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if os.stat('news_data.csv').st_size == 0:
            writer.writerow(['id', 'title', 'text', 'label'])

        for idx, (title, content) in enumerate(news_map.items(), len(existing_titles) + 1):
            if title not in existing_titles:
                writer.writerow([idx, title, content, 'REAL'])
                existing_titles.add(title)

        for idx, (title, content) in enumerate(fake_news_map.items(), len(existing_titles) + 1):
            if title not in existing_titles:
                writer.writerow([idx, title, content, 'FAKE'])
                existing_titles.add(title)

    shuffle_csv_file('news_data.csv')


def update_model():
    model_filename = 'model.pkl'
    with open(model_filename, 'rb') as f:
        existing_model = pickle.load(f)
    new_data = pd.read_csv('news_data.csv')
    new_data = new_data.drop(["id", "title"], axis=1)
    tfvect.fit(x_train)
    X_new = tfvect.transform(new_data['text'])
    existing_model.fit(X_new, new_data['label'])
    with open('model.pkl', 'wb') as f:
        pickle.dump(existing_model, f)
    os.remove('news_data.csv')


def fake_news_det(news):
    tfid_x_train = tfvect.fit_transform(x_train)
    tfid_x_test = tfvect.transform(x_test)
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction


def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=scrape_and_save_news, trigger='interval', days=1)  # Run every day
    scheduler.add_job(func=update_model, trigger='interval', days=5)  # Run every 5 days
    scheduler.start()


# Run the scheduler when the Flask application starts
start_scheduler()


@app.route('/predict-with-url', methods=['GET'])
def predictwithurl():
    url = str(request.args['url'])
    article = Article(url)
    article.download()
    article.parse()
    news_text = article.text
    prediction = fake_news_det(news_text)
    return jsonify({'prediction': str(prediction)})


@app.route('/predict', methods=['GET'])
def predict():
    message = str(request.args['message'])
    prediction = fake_news_det(message)
    return jsonify({'prediction': str(prediction)})


if __name__ == '__main__':
    app.run(debug=True)
