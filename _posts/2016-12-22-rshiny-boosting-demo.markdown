---
layout:     post
title:      "Gradient Boosting Demo Using RShiny"
subtitle:   "Play around with model parameters of a Gradient Boosting Machine while fitting a sine wave"
date:       2016-12-22 12:00:00
author:     "Hareesh Bahuleyan"
header-img: "img/img-boosting-shiny.jpg"
---

<link href="https://fonts.googleapis.com/css?family=Raleway" rel="stylesheet">

<style type="text/css">
	p {
	    font-size: 20px;
	    font-family: 'Raleway', sans-serif;
	    text-align: justify;
	}
	
	h2.subheading, li {
	    font-family: 'Raleway', sans-serif;
	}
</style>

I have always wanted to learn R Shiny, but kept postponing it. Finally, I got a chance to go through some <a href="http://shiny.rstudio.com/tutorial/">tutorials</a> on Shiny's web page. It is actually pretty easy to get started if you have basic programming knowledge in R. So, I built this simple app to do a model fitting with gradient boosting while adjusting the parameters in real time. Lets get started then.

## RShiny
The shiny library in R is an easy way to interactively visualize your data analysis. Essentially, one could create a web application (that can be hosted too) and view it on a web browser with minimal or no HTML/CSS/Javascript knowledge requirement. Yes, thats the beauty of shiny, it internally creates the user interface (UI) using these web technologies, while you just need to program in R.

The Shiny framework has primarily two components - the UI component and the server component. In the UI component, we program the front-end where we design input elements like text boxes, sliders, dropdown menus, etc. Once we get the input from the user, we can pass it to the server component to do some processing and then display an output. The output can be rendered in the form of plots, datatables, images, text, etc. In the hands-on section, I will explain what are the inputs and outputs in this application.


## Gradient Boosting
Boosting is an ensemble machine learning technique which combines the predictions from many weak learners. A weak learner is a predictor whose accuracy is just better than chance. An example would be a <a href="https://en.wikipedia.org/wiki/Decision_tree"> decision tree </a> with 1 split - not a strong classifier, but better than flipping a coin to make a prediction. When we combine many such trees by a weighted average of their predictions we end up with a strong classifier. In boosting, the way we decide on individual trees is a process called forward stagewise additive modelling. The trees are learnt sequentially in boosting, with the early learners fitting fairly simple models. As the iteration progresses, the new learners become more complex focusing on the training examples where the previous learners made a mistake or error. At the end of the iteration process, the individual learners are given weights and the final prediction is a linear combination of these individual predictions. The most common weak learners used are decision trees. Check out <a href="https://www.youtube.com/watch?v=sRktKszFmSk"> this</a> video to get a better understanding.   

There are multiple parameters that can be altered while fitting a boosting model. The parameters that I have chosen for my app are the following:

1. <b>Number of trees</b>: Equivalent to the number of iterations as explained above.
2. <b>Interaction Depth</b>: The depth to which each tree should be grown. Depth = 1 implies one split, i.e., 2 nodes.
3. <b>Bagging Fraction</b>: In each iteration, we can use a subset of the training data for fitting the weak learner. Setting a bagging fraction of 0.8 means, randomly sample (without replacement) 80% of the training data for each tree in the sequence. 
4. <b>Shrinkage</b>: This is a kind of regularization parameter, a lower value such as 0.001 implies that we take smaller steps between iterations, penalizing any sudden changes in the model (aiding a slow learning process). Though a lower value can improve the generalizability of the model (due to less over-fitting), it comes at a cost of more computational time. 

## Hands-On
Now that we have a background of the software and the algorithm, lets put it together to design the application. On the UI side, we define panels on which we place the following input elements: 2 sliders and 2 dropdowns. These are the elements with the help of which the user can provide/modify an input. We also need an element to display the output, which is a graph in our case and therefore we define a plot element. The arguments for the input or output elements are pretty straightforward and intuitive. The name that you give for the argument 'inputId' or 'outputId' is the one that would be used to access the element on the server side.

<pre style="background:#fff;color:#000">library(<span style="color:#00a33f">"shiny"</span>)
library(<span style="color:#00a33f">"gbm"</span>)
library(<span style="color:#00a33f">"ggplot2"</span>)

ui <span style="color:#ff5600">&lt;-</span> pageWithSidebar(
  headerPanel(<span style="color:#00a33f">'Select Parameters For GBM'</span>),
  sidebarPanel(
    sliderInput(inputId <span style="color:#ff5600">=</span> <span style="color:#00a33f">"numTrees"</span>, label <span style="color:#ff5600">=</span> <span style="color:#00a33f">"Number of decision trees"</span>, min <span style="color:#ff5600">=</span> 1, max <span style="color:#ff5600">=</span> 200, value <span style="color:#ff5600">=</span> 10),
    selectInput(inputId <span style="color:#ff5600">=</span> <span style="color:#00a33f">"bagFrac"</span>, label <span style="color:#ff5600">=</span> <span style="color:#00a33f">"Sub-sample train data size for each tree"</span>, choices <span style="color:#ff5600">=</span> <span style="color:#ff5600">list</span>(0.5,0.6,0.7,0.8,0.9,<span style="color:#00a33f">"1.0"</span> <span style="color:#ff5600">=</span> 1.0)),
    sliderInput(inputId <span style="color:#ff5600">=</span> <span style="color:#00a33f">"depth"</span>, label <span style="color:#ff5600">=</span> <span style="color:#00a33f">"Depth to which each tree should be grown"</span>, min <span style="color:#ff5600">=</span> 1, max <span style="color:#ff5600">=</span> 5, value <span style="color:#ff5600">=</span> 1),
    selectInput(inputId <span style="color:#ff5600">=</span> <span style="color:#00a33f">"shrinkage"</span>, label <span style="color:#ff5600">=</span> <span style="color:#00a33f">"Shrinkage parameter"</span>, choices <span style="color:#ff5600">=</span> <span style="color:#ff5600">list</span>(1,0.1,0.01,0.001))
  ),
  mainPanel(
    plotOutput(outputId <span style="color:#ff5600">=</span> <span style="color:#00a33f">"predictionPlot"</span>)
  )
)

</pre>

Now that we have our UI ready, lets move on to the server function. For the 'predictionPlot' element defined on UI side, we need to tell shiny to calculate the values and plot the graph. All of that code needs to go into the render function. In this case, I am defining the input variable x as a random uniform distribution and the target variable as sin(x). Then I fit a regression model using gradient boosting, and I make use of all of our inputs in this function (numTrees, bagFrac, depth, shrinkage). Once we have the model, we can go on to predict (Here, I am making the predictions on the training data itself). And the final part is to make a plot between the true values vs the predicted values of the target variable y. I have done this using the famous plotting library in R - <a href="http://docs.ggplot2.org/current/">ggplot2</a>.

<pre style="background:#fff;color:#000"><span style="color:#21439c">server</span> <span style="color:#ff5600">&lt;-</span> <span style="color:#ff5600">function</span>(input, output){
  
  output<span style="color:#ff5600">$</span>predictionPlot <span style="color:#ff5600">&lt;-</span> renderPlot({
    <span style="color:#919191"># Creating the data</span>
    set.seed(100)
    x <span style="color:#ff5600">&lt;-</span> runif(100, min <span style="color:#ff5600">=</span> 0, max <span style="color:#ff5600">=</span> 7)
    x <span style="color:#ff5600">&lt;-</span> sort(x, decreasing <span style="color:#ff5600">=</span> <span style="color:#a535ae">F</span>)
    df <span style="color:#ff5600">&lt;-</span> <span style="color:#ff5600">data.frame</span>(x <span style="color:#ff5600">=</span> x,y <span style="color:#ff5600">=</span> sin(x))
    
    <span style="color:#919191"># Fitting the model</span>
    fit <span style="color:#ff5600">&lt;-</span> gbm(y<span style="color:#ff5600">~</span>x, data=df, distribution=<span style="color:#00a33f">"gaussian"</span>, n.trees <span style="color:#ff5600">=</span> input<span style="color:#ff5600">$</span>numTrees, shrinkage <span style="color:#ff5600">=</span> as.numeric(input<span style="color:#ff5600">$</span>shrinkage), interaction.depth <span style="color:#ff5600">=</span> input<span style="color:#ff5600">$</span>depth, bag.fraction <span style="color:#ff5600">=</span> as.numeric(input<span style="color:#ff5600">$</span>bagFrac))
    <span style="color:#919191"># Make predictions on the train data itself</span>
    predictions <span style="color:#ff5600">&lt;-</span> predict(fit, df, n.trees <span style="color:#ff5600">=</span> input<span style="color:#ff5600">$</span>numTrees)
    df<span style="color:#ff5600">$</span>pred <span style="color:#ff5600">&lt;-</span> predictions
    
    <span style="color:#919191"># Plotting Actual vs Predicted</span>
    ggplot(df, aes(x)) <span style="color:#ff5600">+</span> 
      geom_line(aes(y <span style="color:#ff5600">=</span> y, colour <span style="color:#ff5600">=</span> <span style="color:#00a33f">"Actual"</span>), size=1) <span style="color:#ff5600">+</span> 
      geom_line(aes(y <span style="color:#ff5600">=</span> pred, colour <span style="color:#ff5600">=</span> <span style="color:#00a33f">"Predicted"</span>), size=1) <span style="color:#ff5600">+</span> 
      xlab(<span style="color:#00a33f">"Input Variable (x)"</span>) <span style="color:#ff5600">+</span> ylab(<span style="color:#00a33f">"Output Variable (y)"</span>) <span style="color:#ff5600">+</span>  
      theme(
        axis.title.x <span style="color:#ff5600">=</span> element_text(color=<span style="color:#00a33f">"blue"</span>, size=14, face=<span style="color:#00a33f">"bold"</span>),
        axis.title.y <span style="color:#ff5600">=</span> element_text(color=<span style="color:#00a33f">"maroon"</span>, size=14, face=<span style="color:#00a33f">"bold"</span>),
        axis.text.x <span style="color:#ff5600">=</span> element_text(size=14),
        axis.text.y <span style="color:#ff5600">=</span> element_text(size=14),
        legend.text <span style="color:#ff5600">=</span> element_text(size <span style="color:#ff5600">=</span> 16),
        legend.position <span style="color:#ff5600">=</span> <span style="color:#00a33f">"right"</span>,
        legend.title <span style="color:#ff5600">=</span> element_blank()
      )
    
  }, height <span style="color:#ff5600">=</span> 500, width <span style="color:#ff5600">=</span> 800)
  
}

shinyApp(ui <span style="color:#ff5600">=</span> ui, server <span style="color:#ff5600">=</span> server)
</pre>

So if you have done all the coding part correctly, this is how the application should look like:
<center>
<a href="#">
    <img src="{{ site.baseurl }}/img/Post-5-Boosting-RShiny/app-screenshot.png" alt="Boosting Application">
</a>
</center>
That was pretty simple, right? A beautiful application, up and running in almost no time. The other cool thing is that shiny allows you to publish and share your app with others for free. You just need to sign up on their <a href="https://www.shinyapps.io/"> website</a>. And, this app of mine can be found on <a href="https://hareesh.shinyapps.io/gradient_boosting/"> here</a>.

This was my first attempt with shiny, writing an application with just about 50 lines of code. I am a huge fan of data visualization softwares and will surely try out more apps. And when I do, I ll share my learning here on this blog. 