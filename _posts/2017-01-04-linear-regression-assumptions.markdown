---
layout:     post
title:      "Linear Regression Assumptions"
subtitle:   "Revisiting the assumptions in a linear regression model and looking at the remedial measures"
date:       2017-01-04 12:00:00
author:     "Hareesh Bahuleyan"
background: "/img/post-header.jpg"
---

<link href="https://fonts.googleapis.com/css?family=Raleway:300" rel="stylesheet">

<script type="text/javascript"
   src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

<style type="text/css">
	p {
	    font-size: 17px;
	    font-family: 'Raleway', sans-serif;
	    text-align: justify;
	}
	
	h2.subheading, li {
	    font-family: 'Raleway', sans-serif;
	}
</style>

<p>The linear regression is one of simplest modelling techniques that is widely used. Even though the prediction power of regression models are not as good as many of the recent machine learning techniques, it is very useful in making model interpretations - in answering questions such as which explanatory variables have the most influence on the dependent variable (<script type="math/tex" id="MathJax-Element-11">y</script>). Most of the time, we just dump the variables as input to the software and determine the ‘best-fit line’. As a result, we tend to ignore the underlying assumptions behind a linear regression model, some of which have consequences on the model performance. In this post, I wish to share few of the assumptions that one needs to check for while building a linear regression model. I learnt these during my course on <i>Analytical Techniques in Transportation Engineering</i> taught by <a href="http://www.civil.iitm.ac.in/new/?q=ks_edu">Dr. Karthik K Srinivasan</a> during my undergraduate studies.</p>

<h3 id="1-linear-functional-form">1. Linear Functional Form</h3>

<p>As the name rightly suggests, the y-variable is required to have a linear relationship with each of the <script type="math/tex" id="MathJax-Element-350">x_i</script>. We can visually check this by plotting <script type="math/tex" id="MathJax-Element-351">y</script> against each <script type="math/tex" id="MathJax-Element-352">x_i</script> and see if the scatter plot looks linear. Alternatively, we could also plot the residuals <script type="math/tex" id="MathJax-Element-353">e_i=(y_i-\hat{y}_i)</script> on y-axis and the explanatory variables (<script type="math/tex" id="MathJax-Element-354">x_i</script>) on the x-axis (one at a time). If this gives a horizontal line passing through zero with no particular trend among the scatter points, then the assumption of linearity holds good. Incase the graphs do not come out as mentioned above, then it could be because of a violation of the assumption. In such a scenario the respective <script type="math/tex" id="MathJax-Element-355">x_i</script> can be transformed - For example, one could use or <script type="math/tex" id="MathJax-Element-356">\sqrt{x_i}</script> or <script type="math/tex" id="MathJax-Element-357">\log{x_i}</script> instead of <script type="math/tex" id="MathJax-Element-358">x_i</script>. The figures below show the ideal graphs where this assumption holds true.</p>

<h3 id="2-constant-error-variance">2. Constant Error Variance</h3>

<p>Also known by the term Homoscedasticity, this refers to the fact that the residual <script type="math/tex" id="MathJax-Element-444">e_i</script> should have a near-constant variance when taken across all data points. In the graph below plot between <script type="math/tex" id="MathJax-Element-445">e_i</script> and <script type="math/tex" id="MathJax-Element-446">\hat{y_i}</script>, the variance is low initially and increases later, which is a clear violation of this assumption. A detailed check would involve plots of  <script type="math/tex" id="MathJax-Element-447">\hat{y_i}</script> vs each <script type="math/tex" id="MathJax-Element-448">x_i</script> one-by-one, keeping other <script type="math/tex" id="MathJax-Element-449">x_i</script> constant or at the same level. As example of violation seen through this plot would be as below. <br>
Similar to the previous case, one could try to transform individual <script type="math/tex" id="MathJax-Element-450">x_i</script> variables. Another way to overcome this issue is using weighted least squares for model fitting instead of ordinary least squares. Here, one could assign a lower weight to records that cause more variation. </p>

<h3 id="3-uncorrelated-error-terms">3. Uncorrelated Error Terms</h3>

<p>In a linear regression model, the error terms are assumed to be uncorrelated with each other. A plot of the residuals <script type="math/tex" id="MathJax-Element-457">e_i</script> across data points should have no clear pattern. The graph below (zig-zag pattern) is a clear indication of negative correlation between residuals. However, it may not always be evident from the graph. A more scientific approach is to determine the <a href="https://en.wikipedia.org/wiki/Durbin%E2%80%93Watson_statistic">Durbin–Watson statistic</a> which is calculated as <script type="math/tex" id="MathJax-Element-458">DW = \frac{\sum_{i=1}^N (e_i-e_{i-1})^2}{\sum_{i=1}^N (e_i)^2}</script>, where N stands for the number of data records. A DW-stat value between 1 and 3 is ideal - the assumption is not violated. However, a value &gt; 3 indicates negative correlation and a value &lt; 1 means there exists positive correlation. This is a very difficult problem to correct if encountered. The use of generalized least squares instead of ordinary least squares for coefficient estimation could alleviate this issue to some extent.</p>

<h3 id="4-no-outliers">4. No Outliers</h3>

<p>The presence of outliers in the data can have an impact on the model because it gives a wrong picture of the actual relationship. The slope and intercept (coefficients) may be very different with and without outliers. The quality of the fit measured by <script type="math/tex" id="MathJax-Element-463">R^2</script> also gets affected. <br>
Ideally, it is best to get rid out outliers with the help of boxplots or other methods in the data preparation stage itself. One could also look at how much the outliers impact the model (the coefficients, <script type="math/tex" id="MathJax-Element-464">R^2</script> ). This could be done by fitting the model with and without the outliers.  Additionally, one can use <a href="http://rforpublichealth.blogspot.ca/2013/08/exporting-results-of-linear-regression_24.html">Robust Standard Errors</a> while checking for coefficient significance. This would take into account the fact there are outliers present in the data. The robust standard errors can also be used where we observe that the error terms do not have a constant variance (known as <a href="http://www.statsmakemecry.com/smmctheblog/confusing-stats-terms-explained-heteroscedasticity-heteroske.html">heteroscedasticity</a>).</p>

<h3 id="5-normal-distribution-of-error">5. Normal Distribution of Error</h3>

<p>The linear regression model requires that the residuals (<script type="math/tex" id="MathJax-Element-491">e_i</script>) follow a normal distribution with mean zero. If the cumulative distribution curve of the residuals follow a S-shape (as that of a <a href="https://en.wikipedia.org/wiki/Normal_distribution#/media/File:Normal_Distribution_CDF.svg"> standard normal CDF</a>) or the probability density function (PDF) is approximately <a href="https://en.wikipedia.org/wiki/Normal_distribution#/media/File:Normal_Distribution_PDF.svg"> bell-shaped</a>. However, it may become difficult to distinguish visually. In such a case, it is recommended that you perform a goodness-of-fit test that compares the observed v/s the theoretical values. One could go for either of the <a href="https://en.wikipedia.org/wiki/Chi-squared_test">Chi-squared test </a> or the <a href="https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test">K-S test</a>.  <br>
In case of a violation of this assumption, one could try a variable transformation. For example, <script type="math/tex" id="MathJax-Element-492">x_i</script> may not give normally distributed errors, instead one could try <script type="math/tex" id="MathJax-Element-493">\log{x_i}</script>. In the worst case, you may be required to switch to some other coefficient estimation technique like maximum likelihood estimation (rather than least squares method).</p>

<h3 id="6-absence-of-multi-collinearity">6. Absence of Multi-collinearity</h3>

<p>If the absolute value of the correlation between two explanatory variables <script type="math/tex" id="MathJax-Element-985">|corr(X_1, X_2)|</script> is greater than a threshold (usually 0.4 or 0.5), then it may result in multi-collinearity. In linear regression, it is assumed that the explanatory variables are independent of each other. A second level check for multi-collinearity could be done using the <a href="https://onlinecourses.science.psu.edu/stat501/node/347"> variation inflation factor (VIF)</a>. For this, you will need to build a linear regression model with one of the  <script type="math/tex" id="MathJax-Element-986">x_i</script> as dependent variable regressed with the other <script type="math/tex" id="MathJax-Element-987">x_i</script> as the predictor variables. I’ll give you an example to make things clear. If we originally have 3 explanatory variables <script type="math/tex" id="MathJax-Element-988">x_1, x_2, x_3</script> and you suspect that there is multi-collinearity, then you build a new regression model of the form <script type="math/tex" id="MathJax-Element-989">x_1=\beta_0+\beta_1x_2+\beta_2x_3</script>, and estimate the <script type="math/tex" id="MathJax-Element-990">R^2</script>. With this value of <script type="math/tex" id="MathJax-Element-991">R^2</script>, you can calculate the <script type="math/tex" id="MathJax-Element-992">VIF = \frac{1}{1-R^2}</script>. If the <script type="math/tex" id="MathJax-Element-993">VIF>10</script>, then one can conclude the presence of multi-collinearity, which is undesirable. One solution would be to drop one or more of the predictor variables <script type="math/tex" id="MathJax-Element-994">x_i</script> and re-fit the model. The extend of correlation may decrease on doing some transformations. For example, use <script type="math/tex" id="MathJax-Element-995">\log{x_1}, \log{x_2}</script> instead of <script type="math/tex" id="MathJax-Element-996">x_1, x_2</script> .</p>

<p>So those were the assumptions that you need to consider when using linear regression to model your data.   The consequences of violation can be anything ranging from a poor model fit to discarding variables from the model that were actually significant. The next time you use a linear regression model, make it a point to check whether these assumptions hold, if not, try out some of the remedial measures mentioned. This could ultimately lead to better model performance and would help in more accurate interpretation of the coefficients. </p>