# Sentiment Analysis
Data Science Indy Meetup  
9 Sept 2015  


## Getting and preparing data

In R, you use `read.csv` to read CSV files into `data.frame` variables. Although the R function `read.csv` can work with URLs, https is a problem for R in many cases, so you need to use a package like RCurl to get around it. Moreover, from the Kaggle page description we know that the file is tab-separated, there is not header, and we need to disable quoting since some sentences include quotes and that will stop file parsing at some point.  



```r
setwd("~/Sentiment-Analysis-Showdown/04-sentiment-analysis")
library(RCurl)
```

```
## Loading required package: bitops
```

```r
#test_data_url <- "https://kaggle2.blob.core.windows.net/competitions-data/inclass/2558/testdata.txt?sv=2012-02-12&se=2015-08-06T10%3A32%3A23Z&sr=b&sp=r&sig=a8lqVKO0%2FLjN4hMrFo71sPcnMzltKk1HN8m7OPolArw%3D"
#train_data_url <- "https://kaggle2.blob.core.windows.net/competitions-data/inclass/2558/training.txt?sv=2012-02-12&se=2015-08-06T10%3A34%3A08Z&sr=b&sp=r&sig=meGjVzfSsvayeJiDdKY9S6C9ep7qW8v74M6XzON0YQk%3D"

#test_data_file <- getURL(test_data_url)
#train_data_file <- getURL(train_data_url)

train_data_df <- read.csv("train_data.csv", 
    sep='\t', 
    header=FALSE, 
    quote = "",
    stringsAsFactor=F,
    col.names=c("Sentiment", "Text"))
test_data_df <- read.csv("test_data.csv", 
    sep='\t', 
    header=FALSE, 
    quote = "",
    stringsAsFactor=F,
    col.names=c("Text"))
# we need to convert Sentiment to factor
train_data_df$Sentiment <- as.factor(train_data_df$Sentiment)
```

Now we have our data in data frames. We have 7086 sentences for the training data and 33052 sentences for the test data. The sentences are in a column named `Text` and the sentiment tag (just for training data) in a column named `Sentiment`. Let's have a look at the first few lines of the training data.  


```r
head(train_data_df)
```

```
##   Sentiment
## 1         0
## 2         1
## 3         1
## 4         1
## 5         1
## 6         1
##                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  Text
## 1                                                                                                                                                                                                                                                                                                                                                                                                                                        In the beginning God created the heavens and the earth. Now the earth was formless and empty, darkness was over the surface of the deep, and the Spirit of God was hovering over the waters.
## 2                                                                                                                                                                                                                                                                                                                                                                     And God said, Let there be light, and there was light. God saw that the light was good, and he separated the light from the darkness. God called the light day, and the darkness he called night. And there was evening, and there was morningâ<U+0080><U+0094>the first day.
## 3                                                                                                                                                                                                                                                                                                                                                  And God said, Let there be a vault between the waters to separate water from water. So God made the vault and separated the water under the vault from the water above it. And it was so. God called the vault sky. And there was evening, and there was morningâ<U+0080><U+0094>the second day.
## 4                                                                                                                                                                                                                                                                                                                                                                                                                 And God said, Let the water under the sky be gathered to one place, and let dry ground appear. And it was so. God called the dry ground land, and the gathered waters he called seas. And God saw that it was good.
## 5                                                                                                                                                                                                                Then God said, Let the land produce vegetation: seed-bearing plants and trees on the land that bear fruit with seed in it, according to their various kinds. And it was so. The land produced vegetation: plants bearing seed according to their kinds and trees bearing fruit with seed in it according to their kinds. And God saw that it was good. And there was evening, and there was morningâ<U+0080><U+0094>the third day.
## 6 And God said, Let there be lights in the vault of the sky to separate the day from the night, and let them serve as signs to mark sacred times, and days and years, and let them be lights in the vault of the sky to give light on the earth. And it was so. God made two great lightsâ<U+0080><U+0094>the greater light to govern the day and the lesser light to govern the night. He also made the stars. God set them in the vault of the sky to give light on the earth, to govern the day and the night, and to separate light from darkness. And God saw that it was good. And there was evening, and there was morningâ<U+0080><U+0094>the fourth day.
```

We can also get a glimpse at how tags ar distributed. In R we can use `table`.  


```r
table(train_data_df$Sentiment)
```

```
## 
##     0     1 
## 12740 17719
```

That is, we have data more or less evenly distributed, with 12740 negatively tagged sentences, and 17719 positively tagged sentences. How long on average are our sentences in words?    


```r
mean(sapply(sapply(train_data_df$Text, strsplit, " "), length))
```

```
## [1] 10.04087
```

About 10 words in length.  

## Preparing a corpus  

> In linguistics, a corpus (plural corpora) or text corpus is a large and structured set of texts (nowadays usually electronically stored and processed). They are used to do statistical analysis and hypothesis testing, checking occurrences or validating linguistic rules within a specific language territory.  
> Source: [Wikipedia](https://en.wikipedia.org/wiki/Text_corpus)  

In this section we will process our text sentences and create a corpus. We will also extract important words and stablish them as input variables for our classifier.  


```r
library(tm)
```

```
## Loading required package: NLP
```

```r
corpus <- Corpus(VectorSource(c(train_data_df$Text, test_data_df$Text)))
corpus
```

```
## <<VCorpus>>
## Metadata:  corpus specific: 0, document level (indexed): 0
## Content:  documents: 63000
```

Let's explain what we just did. First we used both, test and train data. We need to consider all possible word in our corpus. Then we created a `VectorSource`, that is the input type for the `Corpus` function defined in the package `tm`. That gives us a `VCorpus` object that basically is a collection of content+metadata objects, where the content contains our sentences. For example, the content on the first document looks like this.    


```r
corpus[1]$content
```

```
## [[1]]
## <<PlainTextDocument>>
## Metadata:  7
## Content:  chars: 188
```

In order to make use of this corpus, we need to transform its contents as follows.  


```r
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, PlainTextDocument)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, stemDocument)
```

First we put everything in lowercase. The second transformation is needed in order to have each document in the format we will need later on. Then we remove punctuation, english stopwords, strip whitespaces, and [stem](https://en.wikipedia.org/wiki/Stemming) each word. Right now, the first entry now looks like this.  


```r
corpus[1]$content
```

```
## [[1]]
## <<PlainTextDocument>>
## Metadata:  7
## Content:  chars: 94
```

In our way to find document input features for our classifier, we want to put this corpus in the shape of a document matrix. A document matrix is a numeric matrix containing a column for each different word in our whole corpus, and a row for each document. A given cell equals to the freqency in a document for a given term.  

This is how we do it in R.  


```r
dtm <- DocumentTermMatrix(corpus)
dtm
```

```
## <<DocumentTermMatrix (documents: 63000, terms: 8564)>>
## Non-/sparse entries: 364129/539167871
## Sparsity           : 100%
## Maximal term length: 614
## Weighting          : term frequency (tf)
```

If we consider each column as a term for our model, we will end up with a very complex model with 8383 different features. This will make the model slow and probably not very efficient. Some terms or words are more important than others, and we want to remove those that are not so much. We can use the function `removeSparseTerms` from the `tm` package where we pass the matrix and a number that gives the maximal allowed sparsity for a term in our corpus. For example, if we want terms that appear in at least 1% of the documents we can do as follows.  


```r
sparse <- removeSparseTerms(dtm, 0.99)
sparse
```

```
## <<DocumentTermMatrix (documents: 63000, terms: 84)>>
## Non-/sparse entries: 182208/5109792
## Sparsity           : 97%
## Maximal term length: 9
## Weighting          : term frequency (tf)
```

We end up with just 84 terms. The closer that value is to 1, the more terms we will have in our `sparse` object, since the number of documents we need a term to be in is smaller.  

Now we want to convert this matrix into a data frame that we can use to train a classifier in the next section.  


```r
important_words_df <- as.data.frame(as.matrix(sparse))
colnames(important_words_df) <- make.names(colnames(important_words_df))
# split into train and test
important_words_train_df <- head(important_words_df, nrow(train_data_df))
important_words_test_df <- tail(important_words_df, nrow(test_data_df))

# Add to original dataframes
train_data_words_df <- cbind(train_data_df, important_words_train_df)
test_data_words_df <- cbind(test_data_df, important_words_test_df)

# Get rid of the original Text field
train_data_words_df$Text <- NULL
test_data_words_df$Text <- NULL
```

Now we are ready to train our first classifier.  

## A bag-of-words linear classifier  

The approach we are using here is called a [bag-of-words model](https://en.wikipedia.org/wiki/Bag-of-words_model). In this kind of model we simplify documents to a multiset of terms frequencies. That means that, for our model, a document sentiment tag will depend on what words appear in that document, discarding any grammar or word order but keeping multiplicity.  

But first of all we need to split our train data into train and test data. Why we do that if we already have a testing set? Simple. The test set from the Kaggle competition doesn't have tags at all (obviously). If we want to asses our model accuracy we need a test set with sentiment tags to compare our results. We will split using `sample.split` from the [`caTools`](https://cran.r-project.org/web/packages/caTools/index.html) package.    


```r
library(caTools)
set.seed(1234)
# first we create an index with 80% True values based on Sentiment
spl <- sample.split(train_data_words_df$Sentiment, .85)
# now we use it to split our data into train and test
eval_train_data_df <- train_data_words_df[spl==T,]
eval_test_data_df <- train_data_words_df[spl==F,]
```

Building linear models is something that is at the very heart of R. Therefore is very easy, and it requires just a single function call.  


```r
log_model <- glm(Sentiment~., data=eval_train_data_df, family=binomial)
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```r
summary(log_model)
```

```
## 
## Call:
## glm(formula = Sentiment ~ ., family = binomial, data = eval_train_data_df)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -8.4904  -0.0030   0.0006   0.0080   5.2171  
## 
## Coefficients:
##               Estimate Std. Error z value Pr(>|z|)    
## (Intercept)    0.41304    0.20205   2.044 0.040930 *  
## aaa            1.84555    0.40763   4.528 5.97e-06 ***
## airlin        -0.45838    0.64922  -0.706 0.480161    
## also           0.04584    0.52063   0.088 0.929836    
## amaz           5.91491    0.76005   7.782 7.12e-15 ***
## angelina       1.02415  189.77446   0.005 0.995694    
## anyway         0.35083    1.34351   0.261 0.793995    
## awesom        15.18877    2.81169   5.402 6.59e-08 ***
## back          -5.21335    0.36187 -14.407  < 2e-16 ***
## balboa        13.95245 1083.05752   0.013 0.989722    
## beauti        12.56654    0.97668  12.867  < 2e-16 ***
## boston         1.28329    0.53049   2.419 0.015561 *  
## brokeback      2.21708    0.80266   2.762 0.005742 ** 
## can           -2.63078    0.62705  -4.195 2.72e-05 ***
## citi          -0.53004    0.77777  -0.681 0.495562    
## code           2.06564    1.60281   1.289 0.197481    
## cool           7.87476    1.87321   4.204 2.62e-05 ***
## cruis         -5.45505    8.99155  -0.607 0.544059    
## dont          -1.86438    0.83593  -2.230 0.025727 *  
## even          -0.29485    0.84793  -0.348 0.728044    
## francisco      1.47033    3.75027   0.392 0.695015    
## friend        -3.32083    0.73891  -4.494 6.98e-06 ***
## fuck          -5.46431    1.77311  -3.082 0.002058 ** 
## geico          3.23327    1.11693   2.895 0.003794 ** 
## get            1.83000    1.01222   1.808 0.070621 .  
## good           6.88268    0.91951   7.485 7.15e-14 ***
## got           -1.48800    0.64358  -2.312 0.020773 *  
## great          2.61211    0.31336   8.336  < 2e-16 ***
## harri          2.20506    4.67273   0.472 0.636999    
## harvard       -1.27754    0.53426  -2.391 0.016793 *  
## hate         -11.21981    0.69154 -16.224  < 2e-16 ***
## hilton         1.27938   35.85051   0.036 0.971532    
## honda          1.96396    0.38317   5.126 2.97e-07 ***
## imposs         5.50582   47.16555   0.117 0.907071    
## ive            2.26806    0.85482   2.653 0.007972 ** 
## joli          -2.42632  189.77669  -0.013 0.989799    
## just          -3.42671    0.89250  -3.839 0.000123 ***
## know          -1.17470    0.55533  -2.115 0.034404 *  
## laker         -0.58091    0.49366  -1.177 0.239297    
## like           4.16531    0.27539  15.125  < 2e-16 ***
## littl          3.88106    0.79705   4.869 1.12e-06 ***
## london        -0.62603    0.32422  -1.931 0.053500 .  
## look          -1.85442    0.82333  -2.252 0.024300 *  
## love          11.32692    0.66524  17.027  < 2e-16 ***
## macbook        1.82122    1.07813   1.689 0.091172 .  
## make           1.51318    0.97558   1.551 0.120886    
## miss          -1.37548    0.47347  -2.905 0.003671 ** 
## mission       -3.34983   47.16540  -0.071 0.943379    
## mit           -1.32615    0.68024  -1.950 0.051232 .  
## mountain      -5.98593    0.73694  -8.123 4.56e-16 ***
## movi          -2.21188    0.42460  -5.209 1.89e-07 ***
## much          -1.19454    0.72483  -1.648 0.099349 .  
## need           0.66620    1.37097   0.486 0.627014    
## new           -0.63843    0.55291  -1.155 0.248224    
## now           -2.00758    0.67817  -2.960 0.003074 ** 
## one           -1.57933    0.40025  -3.946 7.95e-05 ***
## pari           1.52667   35.85075   0.043 0.966033    
## peopl         -0.90433    0.43858  -2.062 0.039214 *  
## potter        -2.17934    4.66622  -0.467 0.640467    
## purdu         -3.75356    0.68530  -5.477 4.32e-08 ***
## realli        -2.82939    0.49606  -5.704 1.17e-08 ***
## right          1.27987    0.95870   1.335 0.181875    
## said           0.50643    0.93587   0.541 0.588419    
## san            0.94120    3.73774   0.252 0.801189    
## say           -3.10855    0.34764  -8.942  < 2e-16 ***
## seattl         1.18909    0.57307   2.075 0.037990 *  
## see           -2.23333    0.59179  -3.774 0.000161 ***
## shanghai       0.04106    0.62545   0.066 0.947658    
## still          1.83173    0.53543   3.421 0.000624 ***
## stori         -0.28833    1.37139  -0.210 0.833477    
## stuff          2.30111    1.71611   1.341 0.179956    
## stupid       -23.35242  271.82358  -0.086 0.931538    
## suck         -11.38860    1.33008  -8.562  < 2e-16 ***
## thing          0.56357    0.57687   0.977 0.328596    
## think         -2.05053    0.73460  -2.791 0.005249 ** 
## though        -0.25253    0.84354  -0.299 0.764656    
## time           0.13612    0.49100   0.277 0.781600    
## tom            1.40227    8.98706   0.156 0.876008    
## toyota        -3.04677    0.36513  -8.344  < 2e-16 ***
## ucla          -1.25140    1.41680  -0.883 0.377097    
## vinci         -2.21778    1.55358  -1.428 0.153426    
## want           8.09438    0.55259  14.648  < 2e-16 ***
## way           -8.64860    0.94441  -9.158  < 2e-16 ***
## well           1.01838    0.58175   1.751 0.080025 .  
## work           1.78797    0.76591   2.334 0.019572 *  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 35196.3  on 25889  degrees of freedom
## Residual deviance:  1489.3  on 25805  degrees of freedom
## AIC: 1659.3
## 
## Number of Fisher Scoring iterations: 19
```

The first parameter is a formula in the form `Output~Input` where the `.` at the input side means to use every single variable but the output one. Then we pass the data frame and `family=binomial` that means we want to use logistic regression.  

The summary function gives us really good insight into the model we just built. The coefficient section lists all the input variables used in the model. A series of asterisks at the very end of them gives us the importance of each one, with `***` being the greatest significance level, and `**` or `*` being also important. These starts relate to the values in `Pr`. for example, we get that the stem `awesom` has a great significance, with a high positive `Estimate` value. That means that a document with that stem is very likely to be tagged with sentiment 1 (positive). We see the oposite case with the stem `hate`. We also see that there are many terms that doesn't seem to have a great significance.    

So let's use our model with the test data.  


```r
log_pred <- predict(log_model, newdata=eval_test_data_df, type="response")
```

The previous `predict` called with `type="response"` will return probabilities (see [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression)). Let's say that we want a .5 threshold for a document to be classified as positive (Sentiment tag equals 1). Then we can calculate accuracy as follows.   


```r
# Calculate accuracy based on prob
table(eval_test_data_df$Sentiment, log_pred>.5)
```

```
##    
##     FALSE TRUE
##   0  1880   31
##   1    14 2644
```

The cases where our model performed properly are given by the diagonal.  


```r
(1880 + 2644) / nrow(eval_test_data_df)
```

```
## [1] 0.990151
```

This is a very good accuracy. It seems that our bag of words approach works nicely with this particular problem.  

We know we don't have tags on the given test dataset. Still we will try something. We will use our model to tag their entries and then get a random sample of entries and visually inspect how are they tagged. We can do this quickly in R as follows.  


```r
log_pred_test <- predict(log_model, newdata=test_data_words_df, type="response")

test_data_df$Sentiment <- log_pred_test>.5
    
set.seed(1234)
spl_test <- sample.split(test_data_df$Sentiment, .0005)
test_data_sample_df <- test_data_df[spl_test==T,]
```

So lest check what has been classified as positive entries.  


```r
test_data_sample_df[test_data_sample_df$Sentiment==T, c('Text')]
```

```
## [1] "And i've managed to cover all of my own food expenses as of thus far by working as Harvard's random task bitch -- doubly freaking sweet("       
## [2] "but the macbook looks so awesome."                                                                                                              
## [3] "A much better experience than I had in San Francisco, with shitty, crowded bars in Mission, which was supposed to be the \" cool \" area of SF."
## [4] "but i love uiuc muuuuuuuch more!"                                                                                                               
## [5] "miss you aaa ~.."                                                                                                                               
## [6] "All san Francisco has are shitty little old apts in this price range..."                                                                        
## [7] "I'm not crazy about HK either, but Shanghai is sounding awesome."                                                                               
## [8] "Ibn ` Ata'Allah al-Iskandari Be among those who fear and respect Allah."                                                                        
## [9] "aaa my beautiful hair.."
```

And negative ones.  


```r
test_data_sample_df[test_data_sample_df$Sentiment==F, c('Text')]
```

```
## [1] "I hate London."                                                                                                                                           
## [2] "Angelina Jolie says that being self-destructive is selfish and you ought to think of the poor, starving, mutilated people all around the world."          
## [3] "I can only imagine, with the bloated egos and arrogant liberalism at Harvard, this effect was magnified..."                                               
## [4] "Stupid UCLA."                                                                                                                                             
## [5] "I hate the US, I hate George W Bush, I hate of course Israel."                                                                                            
## [6] "+ + + Bruce Willis hat den PrÃ<U+0083> Â¤ sident von Kolumbien verÃ<U+0083> Â¤ rgert weil er ganz arrogant das Land wegen der Koks-Dealer mit Terroristen gleichsetzt.."
## [7] "I really, really hate TOM CRUISE."
```

So judge by yourself. Is our classifier doing a good job at all?

