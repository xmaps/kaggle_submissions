# kaggle_submissions

There are three main ways we can improve our submissions:

* Use a better machine learning algorithm.
* Generate better features.
* Combine multiple machine learning algorithms.

Feature engineering is the most important part of any machine learning task, and there are a lot more features we could calculate. However, we also need a way to figure out which features are the best.

One way to accomplish this is to use univariate feature selection. This approach essentially involves reviewing a data set column by column to identify the ones that correlate most closely with what we're trying to predict.

One thing we can do to improve the accuracy of our predictions is ensemble different classifiers. Ensembling means generating predictions based on information from a set of classifiers, instead of just one. In practice, this means that we average their predictions.

Generally speaking, the more diverse the models we ensemble, the higher our accuracy will be. Diversity means that the models generate their results from different columns, or use very different methods to generate predictions. Ensembling a random forest classifier with a decision tree probably won't work extremely well, because they're very similar. On the other hand, ensembling a linear regression with a random forest can yield very good results.

One caveat with ensembling is that the classifiers we use have to be about the same in terms of accuracy. Ensembling one classifier that's much less accurate than the other will probably make the final result worse.
