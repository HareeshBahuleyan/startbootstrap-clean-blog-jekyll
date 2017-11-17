---
layout:     post
title:      "Data Science Jobs (Part I - Web Scraping)"
subtitle:   "In this first part, we learn how to scrape information from HTML Pages with the help of some Python libraries"
date:       2016-11-26 12:00:00
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

In this first blog post of mine, I will take you through a tutorial of how to scrape information on the internet. 'Data Scientist' is a new buzz word in the job market. So, we will be looking at data scientist jobs posted on <a href="https://www.naukri.com/">Naukri</a>, an Indian job search website. I will be coding this in Python and making use of the following libraries: (1)  <a href="https://docs.python.org/2/library/urllib2.html">urllib2</a> and (2) <a href="https://www.crummy.com/software/BeautifulSoup/bs4/doc/">BeautifulSoup</a>. I have taken inspiration from <a href="https://jessesw.com/Data-Science-Skills/">Jesse's blog</a> who has done a similar analysis on <a href="http://indeed.com">indeed.com</a>.

Lets start by reading the HTML page into an string. Next, we create a soup object using BeautifulSoup that forms a tree structure from the HTML, which is easy to parse. In the code below, I have specified the URL that corresponds to the category of data science or machine learning jobs on naukri.com.

<pre style="background: #fff; color: #000;"><span style="color: #8c868f;"># Specify Base URL</span>
base_url <span style="color: #ff7800;">=</span> <span style="color: #409b1c;">'http://www.naukri.com/machine-learning-jobs-'</span>
source <span style="color: #ff7800;">=</span> urllib2.urlopen(base_url).read()
soup <span style="color: #ff7800;">=</span> bs4.BeautifulSoup(source, <span style="color: #409b1c;">"lxml"</span>)
</pre>

The basic idea once we have the soup object is to search for tags corresponding to each element in the HTML. We can look at the HTML source on the browser by Right Click-> Inspect. Inside each page, there are several hyperlinks and we first need to extract the ones that would redirect us to the job description page.
<pre style="background: #fff; color: #000;">all_links <span style="color: #ff7800;">=</span> [link.get(<span style="color: #409b1c;">'href'</span>) <span style="color: #ff7800;">for</span> link <span style="color: #ff7800;">in</span> soup.findAll(<span style="color: #409b1c;">'a'</span>) <span style="color: #ff7800;">if</span> <span style="color: #409b1c;">'job-listings'</span> <span style="color: #ff7800;">in</span> <span style="color: #3b5bb5;">str</span>(link.get(<span style="color: #409b1c;">'href'</span>))]
<span style="color: #ff7800;">print</span> <span style="color: #409b1c;">"Sample job description link:"</span>,all_links[<span style="color: #3b5bb5;">0</span>]
</pre>
<pre style="background: #fff; color: #000;">Sample job description link: https://www.naukri.com/job-listings-Machine-Learning-Scientist-Data-Science-Premium-Jobs-Mumbai-3-to-4-years-261116002864?src=jobsearchDesk&sid=14801799685391&xp=1
</pre>
Now we have a sample job description page (output link above). So we can crawl this page to get the HTML. Below, I shall explain how this is done for a single page, and I 'll provide the code to do it for all the pages corresponding to data science jobs.
<pre style="background: #fff; color: #000;">jd_url <span style="color: #ff7800;">=</span> all_links[<span style="color: #3b5bb5;">0</span>]
jd_source <span style="color: #ff7800;">=</span> urllib2.urlopen(jd_url).read()
jd_soup <span style="color: #ff7800;">=</span> bs4.BeautifulSoup(jd_source,<span style="color: #409b1c;">"lxml"</span>)
</pre>
Next we extract the individual job level attributes like education, salary, skills required, etc. The key here is to identify the correct tag and extract the text enclosed within those tags.
<pre style="background: #fff; color: #000;"><span style="color: #8c868f;"># Job Location</span>
location <span style="color: #ff7800;">=</span> jd_soup.find(<span style="color: #409b1c;">"div"</span>,{<span style="color: #409b1c;">"class"</span>:<span style="color: #409b1c;">"loc"</span>}).getText().strip()
<span style="color: #ff7800;">print</span> location

<span style="color: #8c868f;"># Job Description</span>
jd_text <span style="color: #ff7800;">=</span> jd_soup.find(<span style="color: #409b1c;">"ul"</span>,{<span style="color: #409b1c;">"itemprop"</span>:<span style="color: #409b1c;">"description"</span>}).getText().strip()
<span style="color: #ff7800;">print</span> jd_text

<span style="color: #8c868f;"># Experience Level</span>
experience <span style="color: #ff7800;">=</span> jd_soup.find(<span style="color: #409b1c;">"span"</span>,{<span style="color: #409b1c;">"itemprop"</span>:<span style="color: #409b1c;">"experienceRequirements"</span>}).getText().strip()
<span style="color: #ff7800;">print</span> experience

<span style="color: #8c868f;"># Role Level Information</span>
labels <span style="color: #ff7800;">=</span> [<span style="color: #409b1c;">'Salary'</span>, <span style="color: #409b1c;">'Industry'</span>, <span style="color: #409b1c;">'Functional Area'</span>, <span style="color: #409b1c;">'Role Category'</span>, <span style="color: #409b1c;">'Design Role'</span>]
role_info <span style="color: #ff7800;">=</span> [content.getText().split(<span style="color: #409b1c;">':'</span>)[<span style="color: #ff7800;">-</span><span style="color: #3b5bb5;">1</span>].strip() <span style="color: #ff7800;">for</span> content <span style="color: #ff7800;">in</span> jd_soup.find(<span style="color: #409b1c;">"div"</span>,{<span style="color: #409b1c;">"class"</span>:<span style="color: #409b1c;">"jDisc mt20"</span>}).contents 
 <span style="color: #ff7800;">if</span> <span style="color: #3b5bb5;">len</span>(<span style="color: #3b5bb5;">str</span>(content).replace(<span style="color: #409b1c;">' '</span>,<span style="color: #409b1c;">''</span>))<span style="color: #ff7800;">!=</span><span style="color: #3b5bb5;">0</span>]

role_info_dict <span style="color: #ff7800;">=</span> {label: role_info <span style="color: #ff7800;">for</span> label, role_info <span style="color: #ff7800;">in</span> <span style="color: #3b5bb5;">zip</span>(labels, role_info)}
<span style="color: #ff7800;">print</span> role_info_dict

<span style="color: #8c868f;"># Skills required</span>
key_skills <span style="color: #ff7800;">=</span> <span style="color: #409b1c;">'|'</span>.join(jd_soup.find(<span style="color: #409b1c;">"div"</span>,{<span style="color: #409b1c;">"class"</span>:<span style="color: #409b1c;">"ksTags"</span>}).getText().split(<span style="color: #409b1c;">'  '</span>))[<span style="color: #3b5bb5;">1</span>:]
<span style="color: #ff7800;">print</span> key_skills

<span style="color: #8c868f;"># Education Level</span>
edu_info <span style="color: #ff7800;">=</span> [content.getText().split(<span style="color: #409b1c;">':'</span>) <span style="color: #ff7800;">for</span> content <span style="color: #ff7800;">in</span> jd_soup.find(<span style="color: #409b1c;">"div"</span>,{<span style="color: #409b1c;">"itemprop"</span>:<span style="color: #409b1c;">"educationRequirements"</span>}).contents 
 <span style="color: #ff7800;">if</span> <span style="color: #3b5bb5;">len</span>(<span style="color: #3b5bb5;">str</span>(content).replace(<span style="color: #409b1c;">' '</span>,<span style="color: #409b1c;">''</span>))<span style="color: #ff7800;">!=</span><span style="color: #3b5bb5;">0</span>]

edu_info_dict <span style="color: #ff7800;">=</span> {label.strip(): edu_info.strip() <span style="color: #ff7800;">for</span> label, edu_info <span style="color: #ff7800;">in</span> edu_info}

<span style="color: #8c868f;"># Sometimes the education information for one of the degrees can be missing</span>
edu_labels <span style="color: #ff7800;">=</span> [<span style="color: #409b1c;">'UG'</span>, <span style="color: #409b1c;">'PG'</span>, <span style="color: #409b1c;">'Doctorate'</span>]
<span style="color: #ff7800;">for</span> l <span style="color: #ff7800;">in</span> edu_labels:
    <span style="color: #ff7800;">if</span> l <span style="color: #ff7800;">not</span> <span style="color: #ff7800;">in</span> edu_info_dict.keys():
        edu_info_dict[l] <span style="color: #ff7800;">=</span> <span style="color: #409b1c;">''</span>
<span style="color: #ff7800;">print</span> edu_info_dict

<span style="color: #8c868f;"># Company Info</span>
company_name <span style="color: #ff7800;">=</span> jd_soup.find(<span style="color: #409b1c;">"div"</span>,{<span style="color: #409b1c;">"itemprop"</span>:<span style="color: #409b1c;">"hiringOrganization"</span>}).contents[<span style="color: #3b5bb5;">1</span>].p.getText()
<span style="color: #ff7800;">print</span> company_name
</pre>
The attributes for the sample job description are shown in the output below:
<pre style="background: #fff; color: #000;">Mumbai

We are looking for a machine learning scientist who can use their skills to research, build and implement solutions in the field of natural language processing, automated answers, semantic knowledge extraction from structured data and unstructured text. You should have a deep love for Machine Learning, Natural Language processing and a strong desire to solve challenging problems.  Responsibilities :  - Using NLP and machine learning techniques to create scalable solutions.  - Researching and coming up with novel approaches to solve real world problems.  - Working closely with the engineering teams to drive real-time model implementations and new feature creations.

3 - 4 yrs

{'Salary': u'Not Disclosed by Recruiter', 'Functional Area': u'Analytics & Business Intelligence', 'Industry': u'IT-Software  /    Software Services', 'Role Category': u'Analytics & BI', 'Design Role': u'Data Analyst'}

Machine Learning|Natural Language Processing|NLP|Research|Statistical Models|Big data|Statistical Modeling

{u'UG': u'Any Graduate - Any Specialization', u'Doctorate': u'Doctorate Not Required', 'PG': ''}

Premium-Jobs

</pre>
Each job posted on this website has job level attributes as shown above. So it makes sense to store them in a table like structure. The dataframe object from Pandas library is very handy in such a scenario. I will show you how we can create one with the information above.
<pre style="background: #fff; color: #000;"><span style="color: #ff7800;">import</span> pandas <span style="color: #ff7800;">as</span> pd
<span style="color: #ff7800;">from</span> pandas <span style="color: #ff7800;">import</span> DataFrame
naukri_df <span style="color: #ff7800;">=</span> pd.DataFrame()
column_names <span style="color: #ff7800;">=</span> [<span style="color: #409b1c;">'Location'</span>, <span style="color: #409b1c;">'Link'</span>, <span style="color: #409b1c;">'Job Description'</span>, <span style="color: #409b1c;">'Experience'</span>,<span style="color: #409b1c;">'Salary'</span>, <span style="color: #409b1c;">'Industry'</span>, <span style="color: #409b1c;">'Functional Area'</span>, <span style="color: #409b1c;">'Role Category'</span>, 
                <span style="color: #409b1c;">'Design Role'</span>, <span style="color: #409b1c;">'Skills'</span>,<span style="color: #409b1c;">'Company Name'</span>, 
                <span style="color: #409b1c;">'UG'</span>,<span style="color: #409b1c;">'PG'</span>,<span style="color: #409b1c;">'Doctorate'</span>]

<span style="color: #ff7800;">from</span> collections <span style="color: #ff7800;">import</span> OrderedDict
df_dict <span style="color: #ff7800;">=</span> OrderedDict({<span style="color: #409b1c;">'Location'</span>:location, <span style="color: #409b1c;">'Link'</span>:all_links[<span style="color: #3b5bb5;">0</span>],<span style="color: #409b1c;">'Job Description'</span>:jd_text,<span style="color: #409b1c;">'Experience'</span>:experience,
                       <span style="color: #409b1c;">'Skills'</span>:key_skills,<span style="color: #409b1c;">'Company Name'</span>:company_name})
df_dict.update(role_info_dict)
df_dict.update(edu_info_dict)

naukri_df <span style="color: #ff7800;">=</span> naukri_df.append(df_dict,ignore_index<span style="color: #ff7800;">=</span><span style="color: #3b5bb5;">True</span>)

<span style="color: #8c868f;"># Reordering the columns to a preferred order as specified</span>
naukri_df <span style="color: #ff7800;">=</span> naukri_df.reindex(columns<span style="color: #ff7800;">=</span>column_names)

<span style="color: #ff7800;">print</span> naukri_df

</pre>

<a href="#">
    <img src="{{ site.baseurl }}/img/Post-1-Web_Scraping/df-image.png" alt="Dataframe Structure">
</a>

We need to do this for all the data science job postings on the website. Lets first check the total number of machine learning jobs posted on the site. This information is present within the tag on the top of the page: div class="count".

<pre style="background:#fff;color:#000">num_jobs <span style="color:#ff7800">=</span> <span style="color:#3b5bb5">int</span>(soup.find(<span style="color:#409b1c">"div"</span>, { <span style="color:#409b1c">"class"</span> : <span style="color:#409b1c">"count"</span> }).h1.contents[<span style="color:#3b5bb5">1</span>].getText().split(<span style="color:#409b1c">' '</span>)[<span style="color:#ff7800">-</span><span style="color:#3b5bb5">1</span>])
<span style="color:#ff7800">print</span> num_jobs
</pre>
<pre style="background:#fff;color:#000">2890
</pre>

So there 2890 job posting (at the time this blog was being written!). And each page has 50 jobs, which means the number of 'listings' pages that we need to crawl to eventually get the links to the final description pages are 58 (Calculated in the code snippet below).  

<pre style="background:#fff;color:#000"><span style="color:#ff7800">import</span> math
num_pages <span style="color:#ff7800">=</span> <span style="color:#3b5bb5">int</span>(math.ceil(num_jobs<span style="color:#ff7800">/</span><span style="color:#3b5bb5">50.0</span>))
<span style="color:#ff7800">print</span> <span style="color:#409b1c">"URL of the last page to be scraped:"</span>, base_url <span style="color:#ff7800">+</span> <span style="color:#3b5bb5">str</span>(num_pages)
</pre>

<pre style="background:#fff;color:#000">URL of the last page to be scraped: http://www.naukri.com/machine-learning-jobs-58
</pre>

Now we need to put all of the above bits and pieces into one single function to extract info about all ML jobs on the site. I have provided a link to my <a href="https://github.com/HareeshBahuleyan/naukri-web-scraping/">github page </a>where you can find the code as well as the complete IPython notebook for this blog. Towards the end of the IPython notebook, you would notice that I have used a library called cPickle. This enables us to save the naukri_df dataframe object as a pickle object, which can be directly loaded back into pandas for later use. It took about an hour's time to scrape all of this information using my laptop.

In my next blog post, I shall use the data that we have extracted, to do some data analysis in order to gain some insight about the general data science job market in India. 



