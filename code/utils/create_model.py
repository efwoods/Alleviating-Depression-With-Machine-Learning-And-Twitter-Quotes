# TRAINING A MODEL 
# Defining dictionary containing all emojis with their meanings.
# emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
#           ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
#           ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
#           ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
#           '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
#           '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
#           ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

# # Get quotes
# df = pd.read_csv('./input/quotes-from-goodread/all_quotes.csv')
# df['Quote'] = df['Quote'].apply(lambda x: re.sub("[\“\”]", "", x))
# df['Other Tags'] = df['Other Tags'].apply(lambda x: re.sub("[\'\[\]]", "", x))

# # Detect Text Language
# langs = []
# for text in df['Quote']:
#     try:
#         lang = detect(text)
#         langs.append(lang)
#     except:
#         lang = 'NaN'
#         langs.append(lang)
# df['lang'] = langs

# df['lang'].value_counts().head(10)

# # Using English Quotes
# df_eng = df[df['lang']=='en']
# print(df_eng.shape)
# df_eng['Main Tag'].value_counts()

# # Preprocessing
# # lower case
# df_eng['CleanQuote'] = df_eng['Quote'].apply(lambda x: x.lower())
# df_eng['CleanQuote'].sample(2)

# # remove stopwords and punctuation
# stop_nltk = stopwords.words("english")
# df_eng['CleanQuote'] = df_eng['CleanQuote'].apply(lambda x: drop_stop(x))
# df_eng['CleanQuote'].sample(2)

# # Tokenize
# df_eng['CleanQuote'] = df_eng['CleanQuote'].apply(lambda x: tokenize(x))

# # drop empty rows
# df_eng.drop(df_eng[df_eng['CleanQuote']==""].index, inplace=True)

# # specify feature and target
# X = df_eng['CleanQuote']
# Y = df_eng['Main Tag']

# # Split into train and test sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=48, stratify=Y)
# print(f"X_train : {X_train.shape}\nX_test : {X_test.shape}")

# # Extract features
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(X_train)

# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# # Create a model (SGDClassifier)
# pipeline_sgd = Pipeline([('vect', CountVectorizer()),
#                          ('tfidf', TfidfTransformer()),
#                          ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
#                           alpha=1e-3, random_state=42)),
#                         ])

# model_sgd = pipeline_sgd.fit(X_train, Y_train)

# predict_sgd = model_sgd.predict(X_test)

# print(classification_report(predict_sgd, Y_test))

# filename = 'model_sgd.joblib'
# joblib.dump(model_sgd, filename)