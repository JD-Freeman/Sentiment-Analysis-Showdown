## Introduction and Getting Data refresh  

## Questions we want to answer  

In any data analysis process, there is one or more questions we want to answer. That is the most basic and important step in the whole process, to define these questions. Since we are going to perform some Exploratory Data Analysis in our TB dataset, these are the questions we want to answer:  

- Which are the countries with the highest and infectious TB incidence?  
- What is the general world tendency in the period from 1990 to 2007?  
- What countries don't follow that tendency?  
- What events might have defined that world tendency and why do we have countries out of tendency?  


## Descriptive Statistics  

### Python  

### R  

## Plotting  

In this section we will take a look at the basic plotting functionality in Python/Pandas and R. However, there are more powerful alternatives like [**ggplot2**](http://ggplot2.org/) that, although originally created for R, has its own [implementation for Python](http://ggplot.yhathq.com/) from the [Yhat](https://yhathq.com/) guys.  

### Python  

Pandas DataFrames implement up to three plotting methods out of the box (check the [documentation](http://pandas.pydata.org/pandas-docs/stable/api.html#id11)). The first one is a basic line plot for each of the series we include in the indexing.  

```python
 existing_df[['United Kingdom', 'Spain', 'Colombia']].plot()
```

![enter image description here](https://www.filepicker.io/api/file/1d39RktiTBKjxFinIjbc "enter image title here")

Or we can use box plots to obtain a summarised view of a given series as follows.

```python
 existing_df[['United Kingdom', 'Spain', 'Colombia']].boxplot()
```
![enter image description here](https://www.filepicker.io/api/file/sT3RjWQQtS9DuSgcNOSi "enter image title here")

There is also a `histogram()` method, but we can't use it with this type of data right now. 

### R  

## Answering Questions  

Let's start with the real fun. Once we know our tools (from the previous tutorial about data frames and this one), let's use them to answer some questions about the incidence and prevalence of infectious tuberculosis in the world.   

**Question**: *We want to know, per year, what country has the highest number of existing and new TB cases.*

### Python  

If we want just the top ones we can make use of `apply` and `argmax`. Remember that, be default, `apply` works with columns (the countries in our case), and we want to apply it to each year. Therefore we need to transpose the data frame before using it, or we can pass the argument `axis=1`.

```python
 existing_df.apply(argmax, axis=1)
```

```python
    year
    1990            Djibouti
    1991            Djibouti
    1992            Djibouti
    1993            Djibouti
    1994            Djibouti
    1995            Djibouti
    1996            Kiribati
    1997            Kiribati
    1998            Cambodia
    1999    Korea, Dem. Rep.
    2000            Djibouti
    2001           Swaziland
    2002            Djibouti
    2003            Djibouti
    2004            Djibouti
    2005            Djibouti
    2006            Djibouti
    2007            Djibouti
    dtype: object
```

But this is too simplistic. Instead, we want to get those countries that are in the fourth quartile. But first we need to find out the world general tendency.

###### World trends in TB cases  


In order to explore the world general tendency, we need to sum up every countries' values for the three datasets, per year.

```python
deaths_total_per_year_df = deaths_df.sum(axis=1)
existing_total_per_year_df = existing_df.sum(axis=1)
new_total_per_year_df = new_df.sum(axis=1)
```

Now we will create a new `DataFrame` with each sum in a series that we will plot using the data frame `plot()` method.

```python
world_trends_df = pd.DataFrame({
           'Total deaths per 100K' : deaths_total_per_year_df, 
           'Total existing cases per 100K' : existing_total_per_year_df, 
           'Total new cases per 100K' : new_total_per_year_df}, 
       index=deaths_total_per_year_df.index)
```
```python
world_trends_df.plot(figsize=(12,6)).legend(
    loc='center left', 
    bbox_to_anchor=(1, 0.5))
```
![enter image description here](https://www.filepicker.io/api/file/CUKWAdPcTeul0jhrtx6z "enter image title here")

It seems that the general tendency is for a decrease in the total number of **existing cases** per 100K. However the number of **new cases** has been increasing, although it seems reverting from 2005. So how is possible that the total number of existing cases is decreasing if the total number of new cases has been growing? One of the reasons could be the observed increae in the number of **deaths** per 100K, but the main reason we have to consider is that people recovers form tuberculosis thanks to treatment. The sum of the recovery rate plus the death rate is greater than the new cases rate. In any case, it seems that there are more new cases, but also that we cure them better. We need to improve prevention and epidemics control.      

###### Countries out of tendency

So the previous was the general tendency of the world as a whole. So what countries are out of that tendency (for bad)? In order to find this out, first we need to know the distribution of countries in an average year.

```python
deaths_by_country_mean = deaths_df.mean()
deaths_by_country_mean_summary = deaths_by_country_mean.describe()
existing_by_country_mean = existing_df.mean()
existing_by_country_mean_summary = existing_by_country_mean.describe()
new_by_country_mean = new_df.mean()
new_by_country_mean_summary = new_by_country_mean.describe()
```

We can plot these distributions to have an idea of how the countries are distributed in an average year.

```python
deaths_by_country_mean.order().plot(kind='bar', figsize=(24,6))
```
![enter image description here](https://www.filepicker.io/api/file/r8PqqNwESKmWupnf1pLQ "enter image title here")

We want those countries beyond 1.5 times the inter quartile range (50%). We have these values in:  

```python
deaths_outlier = deaths_by_country_mean_summary['50%']*1.5
existing_outlier = existing_by_country_mean_summary['50%']*1.5
new_outlier = new_by_country_mean_summary['50%']*1.5
```

Now we can use these values to get those countries that, across the period 1990-2007 has been beyond those levels.

```python
# Now compare with the outlier threshold
outlier_countries_by_deaths_index = 
    deaths_by_country_mean > deaths_outlier
outlier_countries_by_existing_index = 
   existing_by_country_mean > existing_outlier
outlier_countries_by_new_index = 
    new_by_country_mean > new_outlier
```

What proportion of countries do we have out of trend? For deaths:

```python
sum(outlier_countries_by_deaths_index)/num_countries
```
```python
    0.39613526570048307
```

For existing cases (prevalence):

```python
 sum(outlier_countries_by_existing_index)/num_countries
```
```python
    0.39613526570048307
```

For new cases (incidence):

```python
 sum(outlier_countries_by_new_index)/num_countries
```
```python
    0.38647342995169082
```

Now we can use these indices to filter our original data frames.

```python
outlier_deaths_df = deaths_df.T[ outlier_countries_by_deaths_index ].T
outlier_existing_df = existing_df.T[ outlier_countries_by_existing_index ].T
outlier_new_df = new_df.T[ outlier_countries_by_new_index ].T
```

This is serious stuff. We have more than one third of the world being outliers on the distribution of existings cases, new cases, and deaths by infectious tuberculosis. But what if we consider an outlier to be 5 times the IQR? Let's repeat the previous process.

```python
deaths_super_outlier = deaths_by_country_mean_summary['50%']*5
existing_super_outlier = existing_by_country_mean_summary['50%']*5
new_super_outlier = new_by_country_mean_summary['50%']*5
    
super_outlier_countries_by_deaths_index = 
    deaths_by_country_mean > deaths_super_outlier
super_outlier_countries_by_existing_index = 
    existing_by_country_mean > existing_super_outlier
super_outlier_countries_by_new_index = 
    new_by_country_mean > new_super_outlier
```

What proportion do we have now?

```python
sum(super_outlier_countries_by_deaths_index)/num_countries
```
```python
    0.21739130434782608
```

Let's get the data frames.

```python
super_outlier_deaths_df = 
    deaths_df.T[ super_outlier_countries_by_deaths_index ].T
super_outlier_existing_df = 
    existing_df.T[ super_outlier_countries_by_existing_index ].T
super_outlier_new_df = 
    new_df.T[ super_outlier_countries_by_new_index ].T
```

Let's concentrate on epidemics control and have a look at the new cases data frame.

```python
super_outlier_new_df
```
| country | Bhutan | Botswana | Cambodia | Congo, Rep. | Cote d'Ivoire | Korea, Dem. Rep. | Djibouti | Kiribati | Lesotho | Malawi | ... | Philippines | Rwanda | Sierra Leone | South Africa | Swaziland | Timor-Leste | Togo | Uganda | Zambia | Zimbabwe |
|---------|--------|----------|----------|-------------|---------------|------------------|----------|----------|---------|--------|-----|-------------|--------|--------------|--------------|-----------|-------------|------|--------|--------|----------|
| year    |        |          |          |             |               |                  |          |          |         |        |     |             |        |              |              |           |             |      |        |        |          |
| 1990    | 540    | 307      | 585      | 169         | 177           | 344              | 582      | 513      | 184     | 258    | ... | 393         | 167    | 207          | 301          | 267       | 322         | 308  | 163    | 297    | 329      |
| 1991    | 516    | 341      | 579      | 188         | 196           | 344              | 594      | 503      | 201     | 286    | ... | 386         | 185    | 220          | 301          | 266       | 322         | 314  | 250    | 349    | 364      |
| 1992    | 492    | 364      | 574      | 200         | 209           | 344              | 606      | 493      | 218     | 314    | ... | 380         | 197    | 233          | 302          | 260       | 322         | 320  | 272    | 411    | 389      |
| 1993    | 470    | 390      | 568      | 215         | 224           | 344              | 618      | 483      | 244     | 343    | ... | 373         | 212    | 248          | 305          | 267       | 322         | 326  | 296    | 460    | 417      |
| 1994    | 449    | 415      | 563      | 229         | 239           | 344              | 630      | 474      | 280     | 373    | ... | 366         | 225    | 263          | 309          | 293       | 322         | 333  | 306    | 501    | 444      |
| 1995    | 428    | 444      | 557      | 245         | 255           | 344              | 642      | 464      | 323     | 390    | ... | 360         | 241    | 279          | 317          | 337       | 322         | 339  | 319    | 536    | 474      |
| 1996    | 409    | 468      | 552      | 258         | 269           | 344              | 655      | 455      | 362     | 389    | ... | 353         | 254    | 297          | 332          | 398       | 322         | 346  | 314    | 554    | 501      |
| 1997    | 391    | 503      | 546      | 277         | 289           | 344              | 668      | 446      | 409     | 401    | ... | 347         | 273    | 315          | 360          | 474       | 322         | 353  | 320    | 576    | 538      |
| 1998    | 373    | 542      | 541      | 299         | 312           | 344              | 681      | 437      | 461     | 412    | ... | 341         | 294    | 334          | 406          | 558       | 322         | 360  | 326    | 583    | 580      |
| 1999    | 356    | 588      | 536      | 324         | 338           | 344              | 695      | 428      | 519     | 417    | ... | 335         | 319    | 355          | 479          | 691       | 322         | 367  | 324    | 603    | 628      |
| 2000    | 340    | 640      | 530      | 353         | 368           | 344              | 708      | 420      | 553     | 425    | ... | 329         | 348    | 377          | 576          | 801       | 322         | 374  | 340    | 602    | 685      |
| 2001    | 325    | 692      | 525      | 382         | 398           | 344              | 722      | 412      | 576     | 414    | ... | 323         | 376    | 400          | 683          | 916       | 322         | 382  | 360    | 627    | 740      |
| 2002    | 310    | 740      | 520      | 408         | 425           | 344              | 737      | 403      | 613     | 416    | ... | 317         | 402    | 425          | 780          | 994       | 322         | 389  | 386    | 632    | 791      |
| 2003    | 296    | 772      | 515      | 425         | 444           | 344              | 751      | 396      | 635     | 410    | ... | 312         | 419    | 451          | 852          | 1075      | 322         | 397  | 396    | 652    | 825      |
| 2004    | 283    | 780      | 510      | 430         | 448           | 344              | 766      | 388      | 643     | 405    | ... | 306         | 423    | 479          | 898          | 1127      | 322         | 405  | 385    | 623    | 834      |
| 2005    | 270    | 770      | 505      | 425         | 443           | 344              | 781      | 380      | 639     | 391    | ... | 301         | 418    | 509          | 925          | 1141      | 322         | 413  | 370    | 588    | 824      |
| 2006    | 258    | 751      | 500      | 414         | 432           | 344              | 797      | 372      | 638     | 368    | ... | 295         | 408    | 540          | 940          | 1169      | 322         | 421  | 350    | 547    | 803      |
| 2007    | 246    | 731      | 495      | 403         | 420           | 344              | 813      | 365      | 637     | 346    | ... | 290         | 397    | 574          | 948          | 1198      | 322         | 429  | 330    | 506    | 782      |

###### 18 rows × 22 columns  


Let's make some plots to get a better imppression.

```python
super_outlier_new_df.plot(figsize=(12,4)).legend(loc='center left', bbox_to_anchor=(1, 0.5))
```
![enter image description here](https://www.filepicker.io/api/file/cJ335ldPQvmhZJweNngB "enter image title here")

We have 22 countries where the number of new cases on an average year is greater than 5 times the median value of the distribution. Let's create a country that represents on average these 22.

```python
average_super_outlier_country = super_outlier_new_df.mean(axis=1)
average_super_outlier_country
```
```python
    year
    1990    314.363636
    1991    330.136364
    1992    340.681818
    1993    352.909091
    1994    365.363636
    1995    379.227273
    1996    390.863636
    1997    408.000000
    1998    427.000000
    1999    451.409091
    2000    476.545455
    2001    502.409091
    2002    525.727273
    2003    543.318182
    2004    548.909091
    2005    546.409091
    2006    540.863636
    2007    535.181818
    dtype: float64
```

Now let's create a country that represents the rest of the world.

```python
avearge_better_world_country = 
    new_df.T[ - super_outlier_countries_by_new_index ].T.mean(axis=1)
avearge_better_world_country
```
```python
    year
    1990    80.751351
    1991    81.216216
    1992    80.681081
    1993    81.470270
    1994    81.832432
    1995    82.681081
    1996    82.589189
    1997    84.497297
    1998    85.189189
    1999    86.232432
    2000    86.378378
    2001    86.551351
    2002    89.848649
    2003    87.778378
    2004    87.978378
    2005    87.086022
    2006    86.559140
    2007    85.605405
    dtype: float64
```

Now let's plot this country with the average world country.

```python
two_world_df = 
    pd.DataFrame({ 
            'Average Better World Country': avearge_better_world_country,
            'Average Outlier Country' : average_super_outlier_country},
        index = new_df.index)
two_world_df.plot(title="Estimated new TB cases per 100K",figsize=(12,8))
```
![enter image description here](https://www.filepicker.io/api/file/TepcmbKSKiqZt6fmLulo "enter image title here")

The increase in new cases tendency is really stronger in the average super outlier country, so stronget that is difficult to percieve that same tendency in the *better world* country. The 90's decade brought a terrible increse in the number of TB cases in those countries. But let's have a look at the exact numbers.

```python
two_world_df.pct_change().plot(title="Percentage change in estimated new TB cases", figsize=(12,8))
```
![enter image description here](https://www.filepicker.io/api/file/AcmGyyhmSfursmmhiH6q "enter image title here")

The deceleration and reversion of that tendency seem to happen at the same time in both average countries, something around 2002? We will try to find out in the next section.

### Googling about events and dates in Tuberculosis

Well, actually we just went straight to [Wikipedia's entry about the disease](https://en.wikipedia.org/wiki/Tuberculosis#Epidemiology). In the epidemics sections we found the following: 

- The total number of tuberculosis cases has been decreasing since 2005, while **new cases** have decreased since 2002.  
 - This is confirmed by our previous analysis.    

- China has achieved particularly dramatic progress, with about an 80% reduction in its TB mortality rate between 1990 and 2010. Let's check it:  

```python
existing_df.China.plot(title="Estimated existing TB cases in China")
```
![enter image description here](https://www.filepicker.io/api/file/DM29iQNVSquV5PxoGbu3 "enter image title here")

- In 2007, the country with the highest estimated incidence rate of TB was Swaziland, with 1,200 cases per 100,000 people.  

```python
new_df.apply(argmax, axis=1)['2007']
```
```python
    'Swaziland'
```

There are many more findings Wikipedia that we can confirm with these or other datasets from Gapmind world. For example, TB and HIV are frequently associated, together with poverty levels. It would be interesting to joind datasets and explore tendencies in each of them. We challenge the reader to give them a try and share with us their findings. 

### Other web pages to explore

Some interesting resources about tuberculosis apart from the Gapminder website: 

- Gates foundation:  
 - http://www.gatesfoundation.org/What-We-Do/Global-Health/Tuberculosis  
 - http://www.gatesfoundation.org/Media-Center/Press-Releases/2007/09/New-Grants-to-Fight-Tuberculosis-Epidemic  