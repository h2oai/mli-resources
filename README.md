# Machine Learning Interpretability (MLI)

Machine learning algorithms create potentially more accurate models than linear models, but any increase in accuracy over more traditional, better-understood, and more easily explainable techniques is not practical for those who must explain their models to regulators or customers. For many decades, the models created by machine learning algorithms were generally taken to be black-boxes. However, a recent flurry of research has introduced credible techniques for interpreting complex, machine-learned models. Materials presented here illustrate applications or adaptations of these techniques for practicing data scientists.

**Want to contribute your own examples?** Just make a pull request.

A [Dockerfile](anaconda_py35_h2o_xgboost_graphviz/Dockerfile) is provided that will construct a container with all necessary dependencies to run the examples here.

#### Practical MLI examples

  * [Decision tree surrogate models](notebooks/dt_surrogate.ipynb)

  * [LIME (practical samples variant)](notebooks/lime.ipynb)

  * [LOCO (NA variant)](notebooks/loco.ipynb)

  * [Partial dependence and individual conditional expectation (ICE)](notebooks/pdp_ice.ipynb)  

  * [Sensitivity analysis](notebooks/sensitivity_analysis.ipynb)

  * [Monotonic models with XGBoost](notebooks/mono_xgboost.ipynb)

#### Installation

**Dockerfile**

A Dockerfile is provided to build a docker container with all necessary packages and dependencies. This is the easiest way to use these examples if you are on Mac OS X, \*nix, or Windows 10. To do so:

  1. Install and start [docker](https://www.docker.com/).

  From a terminal:

  2. Create a directory for the Dockerfile.</br>
  `$ mkdir anaconda_py35_h2o_xgboost_graphviz`

  3. Fetch the Dockerfile from the mli-resources repo.</br>
  `$ curl https://raw.githubusercontent.com/h2oai/mli-resources/master/anaconda_py35_h2o_xgboost_graphviz/Dockerfile > anaconda_py35_h2o_xgboost_graphviz/Dockerfile`

  4. Build a docker container from the Dockefile.</br>
  `$ docker build anaconda_py35_h2o_xgboost_graphviz`

  5. Display image and container IDs.</br>
  `$ docker ps`

  6. Start the docker image and the Jupyter notebook server.</br>
   `$ docker run -i -t -p 8888:8888 <image_id> /bin/bash -c "/opt/conda/bin/conda install jupyter -y --quiet && /opt/conda/bin/jupyter notebook --notebook-dir=/mli-resources --ip='*' --port=8888 --no-browser"`

  7. Copy the sample data into the Docker image. Refer to [GetData.md](data/GetData.md) to obtain datasets needed for notebooks.</br>
  `$ docker cp path/to/train.csv <container_id>:/mli-resources/data`

***

**Manual**

  Install:

  1. Anaconda Python 4.2.0 from the [Anaconda archives](https://repo.continuum.io/archive/).
  2. [Java](https://java.com/download).
  3. The latest stable [h2o](https://www.h2o.ai/download/) Python package.
  4. [Git](https://git-scm.com/downloads).
  5. [XGBoost](https://github.com/dmlc/xgboost) with Python bindings.
  6. [GraphViz](http://www.graphviz.org/).

  From a terminal:

  7. Clone the mli-resources repository with examples.</br>
  `$ git clone https://github.com/h2oai/mli-resources.git`

  8. `$ cd mli-resources`

  9. Copy the sample data into the mli-resources repo directory. Refer to [GetData.md](data/GetData.md) to obtain datasets needed for notebooks.</br>
  `$ cp path/to/train.csv ./data`

  9. Start the Jupyter notebook server.</br>
  `$ jupyter notebook`

#### References

**General**

* [Machine Learning Interpretability with H2O Driverless AI Booklet](https://www.h2o.ai/wp-content/uploads/2017/09/MLI.pdf)</br>
by Patrick Hall, Navdeep Gill, Megan Kurka, Wen Phan, and the H2O.ai team

* [Towards A Rigorous Science of Interpretable Machine Learning](https://arxiv.org/pdf/1702.08608.pdf)</br>
by Finale Doshi-Velez and Been Kim

* [Ideas for Machine Learning Interpretability](https://www.oreilly.com/ideas/ideas-on-interpreting-machine-learning)</br>
by Patrick Hall, Wen Phan, SriSatish Ambati, and the H2O.ai team

* [Fairness, Accountability, and Transparency in Machine Learning (FAT/ML)](https://www.fatml.org/)

***

**Techniques**

* **Partial Dependence**: [*Elements of Statistical Learning*](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12.pdf), Section 10.13</br>
by Trevor Hastie, Rob Tibshirani, and Jerome Friedman

* **LIME**: [“Why Should I Trust You?” Explaining the Predictions of Any Classifier](http://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf)</br>
by Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin

* **LOCO**: [Distribution-Free Predictive Inference for Regression](http://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf)</br>
by Jing Lei, Max G’Sell, Alessandro Rinaldo, Ryan J. Tibshirani, and Larry Wasserman

* **ICE**: [Peeking inside the black box: Visualizing statistical learning with plots of individual conditional expectation](https://arxiv.org/pdf/1309.6392.pdf)</br>
by Alex Goldstein, Adam Kapelnert, Justin Bleich, and Emil Pitkin

* **Surrogate Models**
  * [Extracting tree structured representations of trained networks](https://papers.nips.cc/paper/1152-extracting-tree-structured-representations-of-trained-networks.pdf)</br>
  by Mark W. Craven and Jude W. Shavlik

  * [Interpreting Blackbox Models via Model Extraction](https://arxiv.org/pdf/1705.08504.pdf)</br>
  by Osbert Bastani, Carolyn Kim, and Hamsa Bastani

***

**Notes**

* [Strata Data Conference slides about MLI](notes/strata_mli_sept_17.pdf) </br>
by Patrick Hall, Wen Phan, SriSatish Ambati, and the H2O.ai team

* [Notes from Patrick Hall's data mining class at GWU](https://github.com/jphall663/GWU_data_mining/blob/master/10_model_interpretability/notes/instructor_notes.pdf)
