import os
import json
from datetime import timedelta, datetime, date

from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.services import InstrumentsService, MarketDataService
from tinkoff.invest.schemas import CandleSource
from tinkoff.invest.utils import now
import pandas as pd
import numpy as np

from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, auc, precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier

from catboost import CatBoostRegressor, CatBoostClassifier

from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel

from sklearn.model_selection import train_test_split

READONLY_TOKEN = os.getenv("TOKEN")

# ISS only allows us to get data with a 15-minute lag. We will use Tinkoff API
# This function should return the last 10 1-minute candles for a given instrument from a given point in time.
# We use 1-minute candles because the 10-minute candles are aligned to the start of the day.
# The candles will be used for training the regression model.
def getLastCandles(client, ticker, starttime = now(), HYPERPARAMETER_HISTORY_LENGTH_IN_MINUTES = 10, stocktype="shares"):
    instruments : InstrumentsService = client.instruments
    marketData : MarketDataService = client.market_data
    id = ""
    found = False
    for item in getattr(instruments, stocktype)().instruments:
        if item.ticker == ticker:
            id = item.uid
            found = True
            break
    if not found:
        return None
    try:
      return marketData.get_candles(
              instrument_id=id,
              from_=starttime - timedelta(minutes=HYPERPARAMETER_HISTORY_LENGTH_IN_MINUTES),
              to=starttime - timedelta(minutes=1),
              interval=CandleInterval.CANDLE_INTERVAL_1_MIN,
              candle_source_type=CandleSource.CANDLE_SOURCE_UNSPECIFIED,).candles
    except:
      try:
        return marketData.get_candles(
                instrument_id=id,
                from_=starttime - timedelta(minutes=HYPERPARAMETER_HISTORY_LENGTH_IN_MINUTES),
                to=starttime - timedelta(minutes=1),
                interval=CandleInterval.CANDLE_INTERVAL_1_MIN,
                candle_source_type=CandleSource.CANDLE_SOURCE_UNSPECIFIED,).candles
      except:
          return None

# this returns a Numpy array with the candle data.
# It reserves 3 slots for positive, negative and neutral sentiments from Dostoevsky
# It normalizes price data to account for the fact that different companies have different stock prices
def arrFromCandles(candles, HYPERPARAMETER_HISTORY_LENGTH_IN_MINUTES = 10):
    candle_arr = np.zeros((1, HYPERPARAMETER_HISTORY_LENGTH_IN_MINUTES * 5 + 4))
    ind = 0
    for candle in candles:
        openval = candle.open.units + candle.open.nano / 1000000000
        candle_arr[0, ind:ind+5] = [
            1,
            (candle.close.units + candle.close.nano / 1000000000) / openval,
            (candle.high.units + candle.high.nano / 1000000000) / openval,
            (candle.low.units + candle.low.nano / 1000000000) / openval,
            candle.volume
            ]
        ind += 5
    return candle_arr

# this returns a list of positive, negative and neutral sentiment for an article
def getArticleSentiments(article: str, model):
  results = model.predict(article)[0]
  return [results['positive'], results['negative'], results['neutral']]

#this normalizes the stock price to obtain the price increase over a specified time after a post's release
def getFuturePriceDelta(client, ticker, releasetime, HYPERPARAMETER_FUTURE_PREDICTION_DEPTH_IN_MINUTES = 60, stocktype="shares", getPrice = False):
    instruments : InstrumentsService = client.instruments
    marketData : MarketDataService = client.market_data
    id = ""
    found = False
    for item in getattr(instruments, stocktype)().instruments:
        if item.ticker == ticker:
            id = item.uid
            found = True
            break
    if not found:
        return None
    try:
      candle = marketData.get_candles(
              instrument_id=id,
              from_=releasetime,
              to=releasetime + timedelta(minutes=HYPERPARAMETER_FUTURE_PREDICTION_DEPTH_IN_MINUTES),
              interval=CandleInterval.CANDLE_INTERVAL_1_MIN,
              candle_source_type=CandleSource.CANDLE_SOURCE_UNSPECIFIED,).candles
    except:
      try:
        candle = marketData.get_candles(
              instrument_id=id,
              from_=releasetime,
              to=releasetime + timedelta(minutes=HYPERPARAMETER_FUTURE_PREDICTION_DEPTH_IN_MINUTES),
              interval=CandleInterval.CANDLE_INTERVAL_1_MIN,
              candle_source_type=CandleSource.CANDLE_SOURCE_UNSPECIFIED,).candles
      except:
        return None
    if not getPrice and len(candle) > 0:
      return (candle[-1].close.units + candle[-1].close.nano / 1000000000) / (candle[0].open.units + candle[0].open.nano / 1000000000)
    elif len(candle) > 0:
      return candle[-1].close.units + candle[-1].close.nano / 1000000000
    return None

#this returns a DataFrame with features corresponding to the candles for all tickers referenced in an article and to the article's sentiment
#jsonArticleData is a JSON corresponding to a single message taken from a TG channel's exported history
def processArticle(jsonArticleData, client, sentimentModel, HYPERPARAMETER_HISTORY_LENGTH_IN_MINUTES = 10, HYPERPARAMETER_FUTURE_PREDICTION_DEPTH_IN_MINUTES = 60):
    candidateTickers = []
    articleText = ""
    releaseTime = datetime.fromisoformat(jsonArticleData["date"])
    for entity in jsonArticleData["text_entities"]:
      if entity["type"] == "cashtag":
        candidateTickers.append(entity['text'][1:])
      else:
        articleText += entity['text']
    if articleText == "":
      return None
    articleData = pd.DataFrame(columns = range(HYPERPARAMETER_HISTORY_LENGTH_IN_MINUTES * 5 + 4))
    articleSentiments = getArticleSentiments(articleText, sentimentModel)
    for ticker in candidateTickers:
        candles = getLastCandles(client, ticker, starttime=releaseTime, HYPERPARAMETER_HISTORY_LENGTH_IN_MINUTES=HYPERPARAMETER_HISTORY_LENGTH_IN_MINUTES)
        if candles is not None and len(candles) != HYPERPARAMETER_HISTORY_LENGTH_IN_MINUTES:
            candidateArr = arrFromCandles(candles, HYPERPARAMETER_HISTORY_LENGTH_IN_MINUTES)
            candidateArr[0, -4:-1] = articleSentiments
            futurePrice = getFuturePriceDelta(client, ticker, releaseTime, HYPERPARAMETER_FUTURE_PREDICTION_DEPTH_IN_MINUTES)
            if futurePrice is None:
              continue
            candidateArr[0, -1] = futurePrice
            articleData = pd.concat([pd.DataFrame(candidateArr, columns=articleData.columns), articleData], ignore_index=True)
    return articleData

def getArticleText(jsonArticleData):
  articleText = ""
  for entity in jsonArticleData["text_entities"]:
      articleText += entity['text']
  return articleText

def getDayCandles(client, ticker, stocktype="etfs"):
    instruments : InstrumentsService = client.instruments
    marketData : MarketDataService = client.market_data
    id = ""
    found = False
    for item in getattr(instruments, stocktype)().instruments:
        if item.ticker == ticker:
            id = item.uid
            found = True
            break
    if not found:
        return None
    print(id)
    try:
      return marketData.get_candles(
          instrument_id=id,
          from_= datetime(2023, 4, 1),
          to= datetime(2024, 4, 1),
          interval=CandleInterval.CANDLE_INTERVAL_DAY,
          candle_source_type=CandleSource.CANDLE_SOURCE_UNSPECIFIED,).candles
    except:
      try:
        return marketData.get_candles(
              instrument_id=id,
              from_= datetime(2023, 4, 1),
              to= datetime(2024, 4, 1),
              interval=CandleInterval.CANDLE_INTERVAL_DAY,
              candle_source_type=CandleSource.CANDLE_SOURCE_UNSPECIFIED,).candles
      except:
          return None

channelLog = json.load(open("myInvestments-3mo.json"))["messages"]
regressionYields = np.empty((0, 3))
with Client(READONLY_TOKEN) as client:
  tokenizer = RegexTokenizer()
  sentimentModel = FastTextSocialNetworkModel(tokenizer=tokenizer)
  for HYPERPARAMETER_FUTURE_PREDICTION_DEPTH_IN_MINUTES in [10, 15, 20, 30, 60]:
    for HYPERPARAMETER_HISTORY_LENGTH_IN_MINUTES in range(10, 20):
      articleData = pd.DataFrame(columns = range(HYPERPARAMETER_HISTORY_LENGTH_IN_MINUTES * 5 + 4))
      for article in channelLog:
        articleDF = processArticle(article, client, sentimentModel, HYPERPARAMETER_HISTORY_LENGTH_IN_MINUTES)
        if articleDF is not None:
          articleData = pd.concat([articleDF, articleData], ignore_index=True)
      epsilon = 0.00025 # this is half the broker's commission, which serves as an acceptable margin of error for the regressor
      train, test = train_test_split(articleData, test_size = 0.2)
      train, val = train_test_split(train, test_size = 0.25)
      X_train, y_train = train[range(HYPERPARAMETER_HISTORY_LENGTH_IN_MINUTES * 5 + 3)], train[HYPERPARAMETER_HISTORY_LENGTH_IN_MINUTES * 5 + 3]
      X_val, y_val = val[range(HYPERPARAMETER_HISTORY_LENGTH_IN_MINUTES * 5 + 3)], val[HYPERPARAMETER_HISTORY_LENGTH_IN_MINUTES * 5 + 3]
      X_test, y_test = test[range(HYPERPARAMETER_HISTORY_LENGTH_IN_MINUTES * 5 + 3)], test[HYPERPARAMETER_HISTORY_LENGTH_IN_MINUTES * 5 + 3]

      print(HYPERPARAMETER_FUTURE_PREDICTION_DEPTH_IN_MINUTES, HYPERPARAMETER_HISTORY_LENGTH_IN_MINUTES)
      catboostModel = CatBoostRegressor()
      catboostModel.fit(X_train, y_train, eval_set = (X_val, y_val), verbose=False)
      y_pred = catboostModel.predict(X_test)
      yield_ = 1
      for (profit_pred, profit_real) in zip(y_pred, y_test):
        if profit_pred > 1 + 4 * epsilon: # we want to be profitable after the commission for both trades is deducted
          yield_ *= (profit_real - 4 * epsilon)
      print("catboost", mean_squared_error(y_pred, y_test), "PROFIT:", yield_)

      yield_ = 1
      svrModel = SVR(epsilon = epsilon / 5)
      svrModel.fit(X_train, y_train)
      y_pred = svrModel.predict(X_test)
      for (profit_pred, profit_real) in zip(y_pred, y_test):
        if profit_pred > 1 + 4 * epsilon: # we want to be profitable after the commission for both trades is deducted
          yield_ *= (profit_real - 4 * epsilon)
      print("SVR", mean_squared_error(y_pred, y_test), "PROFIT:", yield_)

      model = SGDRegressor(loss, epsilon=epsilon, alpha=0.01)
      model.fit(X_train, y_train)

      y_pred = model.predict(X_test)

      yield_ = 1
      for (profit_pred, profit_real) in zip(y_pred, y_test):
        if profit_pred > 1 + 4 * epsilon: # we want to be profitable after the commission for both trades is deducted
          yield_ *= (profit_real - 4 * epsilon)
      print("huber", mean_squared_error(y_pred, y_test), "PROFIT:", yield_)

  print("REGRESSION TRAINING AND EVALUATION COMPLETED")
  with open("riaNews-1y.json", "rb") as fileopen:
    channelLog = json.load(fileopen)["messages"]
    workingDate = None
    dayCandles = getDayCandles(client, "TMOS")
    dayClose = 0
    lastid = 0
    for HYPERPARAMETER_ARTICLES_IN_BATCH in range(1, 6):
      articleData = np.empty((0, 3 * HYPERPARAMETER_ARTICLES_IN_BATCH + 2))
      row = np.zeros((1, 3 * HYPERPARAMETER_ARTICLES_IN_BATCH + 2))
      pos = 0
      candleIndex = 0
      for article in channelLog:
        lastid = article["id"]
        text = getArticleText(article)
        if text == "":
          continue
        releaseTime = datetime.fromisoformat(article["date"])
        if getFuturePriceDelta(client, "TMOS", releaseTime, 1, "etfs", True) is not None:
          if workingDate is None:
            workingDate = releaseTime.date
          elif releaseTime.date != workingDate:
            candleIndex += 1
            workingDate = releaseTime.date
          if candleIndex >= len(dayCandles):
            break
          dayClose = dayCandles[candleIndex].close.units + (dayCandles[candleIndex].close.nano / 1000000000)
          sentiment = getArticleSentiments(text, sentimentModel)
          if pos < 3 * HYPERPARAMETER_ARTICLES_IN_BATCH:
            row[0, pos:pos+3] = sentiment
            pos += 3
          else:
            articleData = np.append(articleData, row, axis=0)
            row[0, :-5] = row[0, 3:-2]
            row[0, -5:-2] = sentiment
            row[0, -2] = (getFuturePriceDelta(client, "TMOS", releaseTime, 1, "etfs", True) < dayClose)
            row[0, -1] = dayClose / (getFuturePriceDelta(client, "TMOS", releaseTime, 1, "etfs", True))
      articleData = np.append(articleData, row, axis=0)
      X, y, yield_ = articleData[:, :-2], articleData[:, -2], articleData[:, -1]
      X_train, X_test, y_train, y_test, yield_train, yield_test = train_test_split(X, y, yield_, stratify=y, test_size = 0.2)
      X_train, X_val, y_train, y_val, yield_train, yield_val = train_test_split(X_train, y_train, yield_train, stratify=y_train, test_size = 0.25)
      for k in [1, 3, 5, 7, 9]:
        model = KNeighborsClassifier(k)
        model.fit(X_train, y_train)

        y_scores = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        precision, recall, thresholds = precision_recall_curve(y_test, y_scores[:, 1])
        auc_pr = auc(recall, precision)

        for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
          profit = 0
          volume = 0
          for i in range(y_test.shape[0]):
            if y_scores[i, 1] > threshold:
              profit += yield_test[i] - 1
              volume += 1
          print("KNN", threshold, HYPERPARAMETER_ARTICLES_IN_BATCH, k, profit, volume, auc_pr)

      catboostModel = CatBoostClassifier()
      catboostModel.fit(X_train, y_train, eval_set = (X_val, y_val), verbose=False)

      y_scores = catboostModel.predict_proba(X_test)
      y_pred = catboostModel.predict(X_test)

      precision, recall, thresholds = precision_recall_curve(y_test, y_scores[:, 1])
      auc_pr = auc(recall, precision)

      
      for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
        profit = 0
        volume = 0
        for i in range(y_test.shape[0]):
          if y_scores[i, 1] > threshold:
            profit += yield_test[i] - 1
            volume += 1
        print("CatBoost", threshold, HYPERPARAMETER_ARTICLES_IN_BATCH, profit, volume, auc_pr)