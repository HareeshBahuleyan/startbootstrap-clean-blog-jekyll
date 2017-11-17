---
layout:     post
title:      "Data Science Jobs (Part II - Visualization)"
subtitle:   "Let us make some cool visualizations with the data that was scraped in Part I"
date:       2016-11-29 12:00:00
author:     "Hareesh Bahuleyan"
background: "/img/post-header.jpg"
---

<link href="https://fonts.googleapis.com/css?family=Raleway:300" rel="stylesheet">

<style type="text/css">
	p {
	    font-size: 17px;
	    font-family: 'Raleway', sans-serif;
	    text-align: justify;
	}
	
	h2.subheading {
	    font-family: 'Raleway', sans-serif;
	}
</style>

In <a href="https://hareeshbahuleyan.github.io/blog/2016/11/26/web-scraping-1/">Part I</a> of this tutorial, the data pertaining to data science jobs was crawled from naukri.com. The job postings had several attributes including location, salary, skills required, education qualifications, etc. to name a few. In this post, I make use of this data to gain some insights about the jobs in this sector in India.

We had saved the data after scaping as a pickle object (cPickle library). Lets start by retrieving this data into our workspace. If you don't have the file from Part I, you can download the data from my <a href="https://github.com/HareeshBahuleyan/naukri-web-scraping/">github repository</a> and start.

<pre style="background:#fff;color:#000"><span style="color:#ff5600">import</span> pandas <span style="color:#ff5600">as</span> pd
<span style="color:#ff5600">from</span> pandas <span style="color:#ff5600">import</span> DataFrame
<span style="color:#ff5600">import</span> cPickle <span style="color:#ff5600">as</span> pickle

<span style="color:#ff5600">with</span> <span style="color:#a535ae">open</span>(<span style="color:#00a33f">'naukri_dataframe.pkl'</span>, <span style="color:#00a33f">'r'</span>) <span style="color:#ff5600">as</span> f:
    naukri_df <span style="color:#ff5600">=</span> pickle.load(f) 
</pre>

The dataframe has a structure as below:
<a href="#">
    <img src="{{ site.baseurl }}/img/Post-2-Jobs_Visualization/df-image.png" alt="Dataframe Structure">
</a>

We can first analyze the location column. The question is - which city in India has the most data science job openings? The pandas dataframe function value_counts() can be used to get the number of jobs per city. 

<pre style="background:#fff;color:#000">naukri_df[<span style="color:#00a33f">'Location'</span>].value_counts()[:10]
</pre>

<pre style="background:#fff;color:#000">Bengaluru              626
Mumbai                 190
Hyderabad              143
Pune                    86
Delhi NCR               57
Chennai                 55
Gurgaon                 55
Delhi NCR,  Gurgaon     44
Delhi                   32
Noida                   29
Name: Location, dtype: int64
</pre>

As you would have already noticed from above, we need to group some places into a single category. For example, 'Delhi', 'Noida', 'Gurgaon' into one category 'Delhi NCR'. The other issue that I encountered is that there are some rows with locations mentioned as comma separated values. For example:

<pre style="background:#fff;color:#000"><span style="color:#ff5600">print</span> naukri_df.ix[499,<span style="color:#00a33f">'Location'</span>]
</pre>

<pre style="background:#fff;color:#000">Delhi NCR,  Mumbai,  Bengaluru,  United States (U.S),  Singapore,  Hong Kong,  Chicago
</pre>

To handle the second issue, I split such comma separated location values and determine a list of unique possible job locations. Then, a string is created by concatenating all the records of 'Location' column. Finally, pattern matching is used to count the occurence of each unique city/location in this string. 

<pre style="background:#fff;color:#000"><span style="color:#ff5600">import</span> re
<span style="color:#ff5600">from</span> collections <span style="color:#ff5600">import</span> defaultdict

<span style="color:#919191"># Find unique locations</span>
uniq_locs <span style="color:#ff5600">=</span> <span style="color:#a535ae">set</span>()
<span style="color:#ff5600">for</span> loc <span style="color:#ff5600">in</span> naukri_df[<span style="color:#00a33f">'Location'</span>]:
    uniq_locs <span style="color:#ff5600">=</span> uniq_locs.union(<span style="color:#a535ae">set</span>(loc.split(<span style="color:#00a33f">','</span>)))
    
uniq_locs <span style="color:#ff5600">=</span> <span style="color:#a535ae">set</span>([item.strip() <span style="color:#ff5600">for</span> item <span style="color:#ff5600">in</span> uniq_locs])

<span style="color:#919191"># All locations into a single string for pattern matching</span>
locations_str <span style="color:#ff5600">=</span> <span style="color:#00a33f">'|'</span>.join(naukri_df[<span style="color:#00a33f">'Location'</span>]) 
loc_dict <span style="color:#ff5600">=</span> defaultdict(<span style="color:#a535ae">int</span>)
<span style="color:#ff5600">for</span> loc <span style="color:#ff5600">in</span> uniq_locs:
    loc_dict[loc] <span style="color:#ff5600">=</span> <span style="color:#a535ae">len</span>(re.findall(loc, locations_str))

<span style="color:#919191"># Take the top 10 most frequent job locations</span>
jobs_by_loc <span style="color:#ff5600">=</span> pd.Series(loc_dict).sort_values(ascending <span style="color:#ff5600">=</span> <span style="color:#a535ae">False</span>)[:10]
<span style="color:#ff5600">print</span> jobs_by_loc
</pre>

<pre style="background:#fff;color:#000">Bengaluru                756
Mumbai                   285
Delhi                    200
Hyderabad                182
Delhi NCR                148
Gurgaon                  128
Pune                     121
Chennai                   73
Noida                     43
Bengaluru / Bangalore     23
</pre>

As can be seen, Bangalore has a lion's share of all machine learning jobs. That was expected, right? Bangalore being the major IT hub of India. Now lets come back to the first issue. We need to combine Gurgaon, Noida and Delhi to Delhi NCR and also keep Bengaluru / Bangalore along with Bengaluru. I had to do this manually.

<pre style="background:#fff;color:#000">jobs_by_loc[<span style="color:#00a33f">'Bengaluru'</span>] <span style="color:#ff5600">=</span> jobs_by_loc[<span style="color:#00a33f">'Bengaluru'</span>] <span style="color:#ff5600">+</span> jobs_by_loc[<span style="color:#00a33f">'Bengaluru / Bangalore'</span>] 
jobs_by_loc[<span style="color:#00a33f">'Delhi NCR'</span>] <span style="color:#ff5600">=</span> jobs_by_loc[<span style="color:#00a33f">'Delhi NCR'</span>] <span style="color:#ff5600">+</span> jobs_by_loc[<span style="color:#00a33f">'Delhi'</span>] <span style="color:#ff5600">+</span> jobs_by_loc[<span style="color:#00a33f">'Noida'</span>] <span style="color:#ff5600">+</span> jobs_by_loc[<span style="color:#00a33f">'Gurgaon'</span>] 
jobs_by_loc.drop([<span style="color:#00a33f">'Bengaluru / Bangalore'</span>,<span style="color:#00a33f">'Delhi'</span>,<span style="color:#00a33f">'Noida'</span>,<span style="color:#00a33f">'Gurgaon'</span>], inplace<span style="color:#ff5600">=</span><span style="color:#a535ae">True</span>)
jobs_by_loc.sort_values(ascending <span style="color:#ff5600">=</span> <span style="color:#a535ae">False</span>, inplace<span style="color:#ff5600">=</span><span style="color:#a535ae">True</span>)
<span style="color:#ff5600">print</span> jobs_by_loc
</pre>

<pre style="background:#fff;color:#000">Bengaluru    779
Delhi NCR    519
Mumbai       285
Hyderabad    182
Pune         121
Chennai       73
</pre>

So thats how the stats look after we combine and group cities. Now Delhi NCR is not that far behind! 

Putting these values into charts make it easier to do the comparison. For data visualization, I have used <a href="http://seaborn.pydata.org/examples/index.html">seaborn</a> and <a href="http://matplotlib.org/">matplotlib</a>. Seaborn is an amazing data visualization tool, I highly recommend that you check it out. 

<pre style="background:#fff;color:#000"><span style="color:#ff5600">import</span> seaborn <span style="color:#ff5600">as</span> sns
<span style="color:#ff5600">import</span> matplotlib.pyplot <span style="color:#ff5600">as</span> plt
<span style="color:#ff5600">%</span>matplotlib inline
sns.set_style(<span style="color:#00a33f">"darkgrid"</span>)
</pre>

<pre style="background:#fff;color:#000">bar_plot <span style="color:#ff5600">=</span> sns.barplot(y<span style="color:#ff5600">=</span>jobs_by_loc.index,x<span style="color:#ff5600">=</span>jobs_by_loc.values,
                        palette<span style="color:#ff5600">=</span><span style="color:#00a33f">"muted"</span>,orient <span style="color:#ff5600">=</span> <span style="color:#00a33f">'h'</span>)                        
plt.title(<span style="color:#00a33f">"Machine Learning Jobs by Location"</span>)
plt.show()
</pre>

<a href="#">
    <img src="{{ site.baseurl }}/img/Post-2-Jobs_Visualization/location.png" alt="Jobs by Location">
</a>

Next let us look at the companies who do maximum hiring in this sector. As seen from the plot below, the top blue chip companies companies like Microsoft, Amazon and GE are among the top recruiters. Some recruiters wish to stay confidential and some others do hiring through consultants like Premium-Jobs.
<pre style="background:#fff;color:#000">jobs_by_companies <span style="color:#ff5600">=</span> naukri_df[<span style="color:#00a33f">'Company Name'</span>].value_counts()[:10]
bar_plot <span style="color:#ff5600">=</span> sns.barplot(y<span style="color:#ff5600">=</span>jobs_by_companies.index,x<span style="color:#ff5600">=</span>jobs_by_companies.values,
                        palette<span style="color:#ff5600">=</span><span style="color:#00a33f">"YlGnBu"</span>,orient <span style="color:#ff5600">=</span> <span style="color:#00a33f">'h'</span>)
plt.title(<span style="color:#00a33f">"Machine Learning Jobs by Companies"</span>)
plt.show()
</pre>

<a href="#">
    <img src="{{ site.baseurl }}/img/Post-2-Jobs_Visualization/company.png" alt="Jobs by Recruiters">
</a>

We have a column for salary and one for experience. What can we make of this? Well, we can see how correlated salary with experience. Lets do this through a scatter plot. However, only a small percentage of the recruiters have explicitly provided the salary. I have made use of only those records which give the salary range. I carried out some string operations in Python to clean the data. For example, a record may have a salary range of INR 5,00,000-9,00,000 and experience range of 3-5 years. For plotting purposes, we need a single value and not a range. So, I calculate the mean value, which are INR 7,00,000 and 4 years respectively, in the above example. 

<pre style="background:#fff;color:#000">salary_list <span style="color:#ff5600">=</span> []
exp_list <span style="color:#ff5600">=</span> []
<span style="color:#ff5600">for</span> i <span style="color:#ff5600">in</span> <span style="color:#a535ae">range</span>(<span style="color:#a535ae">len</span>(naukri_df[<span style="color:#00a33f">'Salary'</span>])):
    salary <span style="color:#ff5600">=</span> naukri_df.ix[i, <span style="color:#00a33f">'Salary'</span>]
    exp <span style="color:#ff5600">=</span> naukri_df.ix[i, <span style="color:#00a33f">'Experience'</span>]
    <span style="color:#ff5600">if</span> <span style="color:#00a33f">'INR'</span> <span style="color:#ff5600">in</span> salary:
        salary_list.append((<span style="color:#a535ae">int</span>(re.sub(<span style="color:#00a33f">','</span>,<span style="color:#00a33f">''</span>,salary.split(<span style="color:#00a33f">"-"</span>)[0].split(<span style="color:#00a33f">"  "</span>)[1])) <span style="color:#ff5600">+</span> <span style="color:#a535ae">int</span>(re.sub(<span style="color:#00a33f">','</span>,<span style="color:#00a33f">''</span>,salary.split(<span style="color:#00a33f">"-"</span>)[1].split(<span style="color:#00a33f">" "</span>)[1])))<span style="color:#ff5600">/</span>2.0)
        exp_list.append((<span style="color:#a535ae">int</span>(exp.split(<span style="color:#00a33f">"-"</span>)[0]) <span style="color:#ff5600">+</span> <span style="color:#a535ae">int</span>(exp.split(<span style="color:#00a33f">"-"</span>)[1].split(<span style="color:#00a33f">" "</span>)[1]))<span style="color:#ff5600">/</span>2.0)
    i<span style="color:#ff5600">+=</span>1

plot_data <span style="color:#ff5600">=</span> pd.DataFrame({<span style="color:#00a33f">'Experience'</span>:exp_list,<span style="color:#00a33f">'Salary'</span>:salary_list})

sns.jointplot(x <span style="color:#ff5600">=</span> <span style="color:#00a33f">'Experience'</span>, y <span style="color:#ff5600">=</span> <span style="color:#00a33f">'Salary'</span>, data<span style="color:#ff5600">=</span>plot_data, kind<span style="color:#ff5600">=</span><span style="color:#00a33f">'reg'</span>, color<span style="color:#ff5600">=</span><span style="color:#00a33f">'maroon'</span>)
plt.ylim((0,6000000))
plt.xlim((0,16))
plt.show()
</pre>

<a href="#">
    <img src="{{ site.baseurl }}/img/Post-2-Jobs_Visualization/experience.png" alt="Salary vs Experience">
</a>

As evident, the salary offered increases with experience. The pearson correlation coefficient is 0.65. Candidates with over 12 years are even offered more than INR 30,00,000 per annum, which is pretty interesting. The graph also depicts the histograms of both salary and experience. Both of these variables have distributions which are skewed towards the lower range of values.

Moving on to the educational qualifications required for this job. We have 3 columns at our disposal - UG, PG and Doctorate. I will just be focusing on the column 'Doctorate', which mentions if a PhD is necessay for the job, if yes, any particular specialization that is preferred. I have made use of Python's <a href="http://www.nltk.org/">nltk</a> to tokenize sentences into words.

<pre style="background:#fff;color:#000"><span style="color:#ff5600">import</span> nltk
<span style="color:#ff5600">from</span> nltk.tokenize <span style="color:#ff5600">import</span> word_tokenize

<span style="color:#ff5600">from</span> collections <span style="color:#ff5600">import</span> Counter

tokens <span style="color:#ff5600">=</span> [word_tokenize(item) <span style="color:#ff5600">for</span> item <span style="color:#ff5600">in</span> naukri_df[<span style="color:#00a33f">'Doctorate'</span>] <span style="color:#ff5600">if</span> <span style="color:#00a33f">'Ph.D'</span> <span style="color:#ff5600">in</span> item]
jobs_by_phd <span style="color:#ff5600">=</span> pd.Series(Counter([item <span style="color:#ff5600">for</span> sublist <span style="color:#ff5600">in</span> tokens <span style="color:#ff5600">for</span> item <span style="color:#ff5600">in</span> sublist <span style="color:#ff5600">if</span> <span style="color:#a535ae">len</span>(item) <span style="color:#ff5600">></span> 4])).sort_values(ascending <span style="color:#ff5600">=</span> <span style="color:#a535ae">False</span>)[:8]
bar_plot <span style="color:#ff5600">=</span> sns.barplot(y<span style="color:#ff5600">=</span>jobs_by_phd.index,x<span style="color:#ff5600">=</span>jobs_by_phd.values,
                        palette<span style="color:#ff5600">=</span><span style="color:#00a33f">"BuGn"</span>,orient <span style="color:#ff5600">=</span> <span style="color:#00a33f">'h'</span>)
plt.title(<span style="color:#00a33f">"Machine Learning Jobs PhD Specializations"</span>)
plt.show()
</pre>

Indeed math and computer science are the two most in demand PhD specializations for a data science role. However, only about 10% of the jobs actually ask for a doctorate degree. So you don't need to spend 5 years doing a PhD to find a top data scientist job. Then you may ask, what exactly are the technical skills that companies look for when hiring? To answer this question, I make use of the skills column to plot the following bar chart. Machine Learning, Python, R, Java, Hadoop, SQL are some of the skills that can land you a data science job. 

<pre style="background:#fff;color:#000">skills <span style="color:#ff5600">=</span> pd.Series(Counter(<span style="color:#00a33f">'|'</span>.join(naukri_df[<span style="color:#00a33f">'Skills'</span>]).split(<span style="color:#00a33f">'|'</span>))).sort_values(ascending <span style="color:#ff5600">=</span> <span style="color:#a535ae">False</span>)[:25]
sns.color_palette(<span style="color:#00a33f">"OrRd"</span>, 10)
bar_plot <span style="color:#ff5600">=</span> sns.barplot(y<span style="color:#ff5600">=</span>skills.values,x<span style="color:#ff5600">=</span>skills.index,orient <span style="color:#ff5600">=</span> <span style="color:#00a33f">'v'</span>)
plt.xticks(rotation<span style="color:#ff5600">=</span>90)
plt.title(<span style="color:#00a33f">"Machine Learning In-Demand Skill Sets"</span>)
plt.show()
</pre>

<a href="#">
    <img src="{{ site.baseurl }}/img/Post-2-Jobs_Visualization/skills.png" alt="Technical Skills">
</a>

And next, we have the last visualization for this post. Moving away from the typical charts and plots, let us do something more exciting. Here, I present to you the wordcloud. A WordCloud is essentially a plot of words present in a document, sized according to their frequency of occurence. I have used the wordcloud library which can be found <a href="https://github.com/amueller/word_cloud">here</a>. And the document that I feed into this function is a string created by concatenating all the 'Job Description' values from our table. You can see for yourself the words that are most frequently used in data science role descriptions.

<pre style="background:#fff;color:#000"><span style="color:#ff5600">from</span> wordcloud <span style="color:#ff5600">import</span> WordCloud, STOPWORDS

jd_string <span style="color:#ff5600">=</span> <span style="color:#00a33f">' '</span>.join(naukri_df[<span style="color:#00a33f">'Job Description'</span>])

wordcloud <span style="color:#ff5600">=</span> WordCloud(font_path<span style="color:#ff5600">=</span><span style="color:#00a33f">'/home/hareesh/Github/naukri-web-scraping/Microsoft_Sans_Serif.ttf'</span>,
                          stopwords<span style="color:#ff5600">=</span>STOPWORDS,background_color<span style="color:#ff5600">=</span><span style="color:#00a33f">'white'</span>, height <span style="color:#ff5600">=</span> 1500, width <span style="color:#ff5600">=</span> 2000).generate(jd_string)

plt.figure(figsize<span style="color:#ff5600">=</span>(10,15))
plt.imshow(wordcloud)
plt.axis(<span style="color:#00a33f">'off'</span>)
plt.show()
</pre>

<a href="#">
    <img src="{{ site.baseurl }}/img/Post-2-Jobs_Visualization/wordcloud.png" alt="WordCloud">
</a>

Thats all folks! I hope you have found some of these job market insights useful. All of the data and code can be found on my <a href="https://github.com/HareeshBahuleyan/naukri-web-scraping/">github repository</a> in the form of an IPython notebook.
