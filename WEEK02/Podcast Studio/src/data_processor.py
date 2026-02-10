from newspaper import Article

def fetch_articles(urls):
    """
    Fetch and parse articles from a list of URLs.
    Returns a list of dictionaries: {"url": url, "text": article_text}
    """
    articles = []
    for url in urls:
        try:
            article = Article(url)
            article.download()
            article.parse()
            articles.append({"url": url, "text": article.text})
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    return articles
