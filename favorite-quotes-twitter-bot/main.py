
import base64
import hashlib
import os
import re
import json
import requests
import redis
from requests.auth import AuthBase, HTTPBasicAuth
from requests_oauthlib import OAuth2Session, TokenUpdated
from flask import Flask, request, redirect, session, url_for, render_template
from dotenv import dotenv_values

config = dotenv_values('.env')

# Since you will be using Redis as your database, you will need to get the environment variable from the previous step and save it into a variable named r that can be called whenever we need to access the database. Using an environment variable allows us to be flexible because you will use an internal connection string when you deploy your bot.
r = redis.from_url(config["REDIS_URL"])

# You will need to set a variable for your app to initialize it, as is typical at the start of every Flask app. You can also create a secret key for your app, so it’s a random string using the os package.
app = Flask(__name__)
app.secret_key = os.urandom(50)

# Back in your Python file, you can set up variables to get your environment variables for your client_id and client_secret. Additionally, you’ll need to define variables for the authorization URL as auth_url and the URL for obtaining your OAuth 2.0 token as token_url. You will also want to get the environment variable you set for your redirect URI and pass that into a new variable called redirect_uri.
client_id = config["CLIENT_ID"]
client_secret = config["CLIENT_SECRET"]
auth_url = "https://twitter.com/i/oauth2/authorize"
token_url = "https://api.twitter.com/2/oauth2/token"
redirect_uri = config["REDIRECT_URI"]

# Now we can set the permissions you need for your bot by defining scopes. You can use the authentication mapping guide to determine what scopes you need based on your endpoints. 
scopes = ["tweet.read", "users.read", "tweet.write", "offline.access"]

# Since Twitter’s implementation of OAuth 2.0 is PKCE-compliant, you will need to set a code verifier. This is a secure random string. This code verifier is also used to create the code challenge.
code_verifier = base64.urlsafe_b64encode(os.urandom(30)).decode("utf-8")
code_verifier = re.sub("[^a-zA-Z0-9]+", "", code_verifier)

# In addition to a code verifier, you will also need to pass a code challenge. The code challenge is a base64 encoded string of the SHA256 hash of the code verifier.
code_challenge = hashlib.sha256(code_verifier.encode("utf-8")).digest()
code_challenge = base64.urlsafe_b64encode(code_challenge).decode("utf-8")
code_challenge = code_challenge.replace("=", "")


# To connect to manage Tweets endpoint, you’ll need an access token. To create this access token, you can create a function called make_token which will pass in the needed parameters and return a token.
def make_token():
    return OAuth2Session(client_id, redirect_uri=redirect_uri, scope=scopes)

# Since your bot will Tweet random facts about cats, you will need to get these from somewhere. There is a cat fact API that you can call to get facts to Tweet. The function parse_cat_fact allows you to make a GET request to the cat fact endpoint and format the JSON response to get a fact you can later Tweet.
def parse_cat_fact():
    url = "https://catfact.ninja/fact"
    cat_fact = requests.request("GET", url).json()
    return cat_fact["fact"]

# To Tweet the cat fact, you can make a function that will indicate it is Tweeting which helps debug and makes a POST request to the Manage Tweets endpoint.
def post_tweet(payload, token):
    print("Tweeting!")
    return requests.request(
        "POST",
        "https://api.twitter.com/2/tweets",
        json=payload,
        headers={
            "Authorization": "Bearer {}".format(token["access_token"]),
            "Content-Type": "application/json",
        },
    )

# At this point, you’ll want to set up the landing page for your bot to authenticate. Your bot will log into a page that lists the permissions needed.
@app.route("/")
def demo():
    global twitter
    twitter = make_token()
    authorization_url, state = twitter.authorization_url(
        auth_url, code_challenge=code_challenge, code_challenge_method="S256"
    )
    session["oauth_state"] = state
    return redirect(authorization_url)


# After the account gives permission to your App you can get the access token. You can format your token to save it as a JSON object into a Redis key/value store so that you can refresh the token the next time your bot Tweets. 

# After you save the token, you can parse the cat fact using the function parse_cat_fact. You will also need to format the cat_fact  into a JSON object. After, you can pass the payload in as a payload into your post_tweet.
@app.route("/oauth/callback", methods=["GET"])
def callback():
    code = request.args.get("code")
    token = twitter.fetch_token(
        token_url=token_url,
        client_secret=client_secret,
        code_verifier=code_verifier,
        code=code,
    )
    st_token = '"{}"'.format(token)
    j_token = json.loads(st_token)
    r.set("token", j_token)
    cat_fact = parse_cat_fact()
    payload = {"text": "{}".format(cat_fact)}
    response = post_tweet(payload, token).json()
    return response