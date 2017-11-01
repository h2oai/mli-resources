# Machine Learning Interpretability(MLI)

Machine learning algorithms create potentially more accurate models than linear models, but any increase in accuracy over more traditional, better-understood, and more easily explainable techniques is not practical for those who must explain their models to regulators or customers. For many decades, the models created by machine learning algorithms were taken to be black-boxes. However, a recent flurry of research has introduced credible techniques for interpreting complex, machine-learned models. Materials presented here illustrate applications or adaptations of these techniques for practicing data scientists.

#### Notes

* [Strata conference slides about MLI](notes/h2o_oreilly_ai_slides.pdf) </br>by Patrick Hall, Wen Phan, SriSatish Ambati, & H2O.ai team

* [Notes from Patrick Hall, Senior Director for Data Science Products at H2o.ai](https://github.com/jphall663/GWU_data_mining/blob/master/10_model_interpretability/notes/instructor_notes.pdf)

* Practical MLI examples (Refer to [GetData.md](data/GetData.md) to obtain datasets needed for notebooks)

  * [Decision tree surrogate models](notebooks/dt_surrogate.ipynb)

  * [LIME (practical samples variant)](notebooks/lime.ipynb)

  * [LOCO (NA variant)](notebooks/loco.ipynb)

  * [Partial dependence and individual conditional expectation (ICE)](notebooks/pdp_ice.ipynb)  

  * [Sensitivity analysis](notebooks/sensitivity_analysis.ipynb)

  * [Monotonic models with XGBoost](notebooks/mono_xgboost.ipynb)

#### References

**General**

* [Machine Learning Interpretability with H2O Driverless AI Booklet](https://www.h2o.ai/wp-content/uploads/2017/09/MLI.pdf)</br>
by Patrick Hall, Navdeep Gill, Megan Kurka & Wen Phan
* [Towards A Rigorous Science of Interpretable Machine Learning](https://arxiv.org/pdf/1702.08608.pdf)</br>
by Finale Doshi-Velez and Been Kim
* [Ideas for Machine Learning Interpretability](https://www.oreilly.com/ideas/ideas-on-interpreting-machine-learning)</br>
by Patrick Hall, Wen Phan, & SriSatish Ambati

***

**Techniques**

* **Partial Dependence**: *Elements of Statistical Learning*, Section 10.13
* **LIME**: [“Why Should I Trust You?” Explaining the Predictions of Any Classifier](http://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf)</br>
by Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin
* **LOCO**: [Distribution-Free Predictive Inference for Regression](http://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf)</br>
by Jing Lei, Max G’Sell, Alessandro Rinaldo, Ryan J. Tibshirani, and Larry Wasserman
* **ICE**: [Peeking inside the black box: Visualizing statistical learning with plots of individual conditional expectation](https://arxiv.org/pdf/1309.6392.pdf)
* **Surrogate Models**
  * [Extracting tree structured representations of trained networks](https://papers.nips.cc/paper/1152-extracting-tree-structured-representations-of-trained-networks.pdf)
  * [Interpreting Blackbox Models via Model Extraction](https://arxiv.org/pdf/1705.08504.pdf)
