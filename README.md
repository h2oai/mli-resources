# Machine Learning Interpretability (MLI)

Machine learning algorithms create potentially more accurate models than linear models, but any increase in accuracy over more traditional, better-understood, and more easily explainable techniques is not practical for those who must explain their models to regulators or customers. For many decades, the models created by machine learning algorithms were generally taken to be black-boxes. However, a recent flurry of research has introduced credible techniques for interpreting complex, machine-learned models. Materials presented here illustrate applications or adaptations of these techniques for practicing data scientists.

**Want to contribute your own examples?** Just make a pull request.

### Contents



A [Dockerfile](anaconda_py35_h2o_xgboost_graphviz/Dockerfile) is provided that will construct a container with all necessary dependencies to run the examples here.

### Practical MLI examples

  * [Decision tree surrogate models](notebooks/dt_surrogate.ipynb)
  * [LIME (practical samples variant)](notebooks/lime.ipynb)
  * [LOCO (NA variant)](notebooks/loco.ipynb)
  * [Partial dependence and individual conditional expectation (ICE)](notebooks/pdp_ice.ipynb)  
  * [Sensitivity analysis](notebooks/sensitivity_analysis.ipynb)
  * [Monotonic models with XGBoost](notebooks/mono_xgboost.ipynb)

#### Installation of Examples

##### Dockerfile

A Dockerfile is provided to build a docker container with all necessary packages and dependencies. This is the easiest way to use these examples if you are on Mac OS X, \*nix, or Windows 10. To do so:

  1. Install and start [docker](https://www.docker.com/).
  From a terminal:
  2. Create a directory for the Dockerfile.</br>
  `$ mkdir anaconda_py35_h2o_xgboost_graphviz`
  3. Fetch the Dockerfile from the mli-resources repo.</br>
  `$ curl https://raw.githubusercontent.com/h2oai/mli-resources/master/anaconda_py35_h2o_xgboost_graphviz/Dockerfile > anaconda_py35_h2o_xgboost_graphviz/Dockerfile`
  4. Build a docker image from the Dockefile.</br>
  `$ docker build anaconda_py35_h2o_xgboost_graphviz`
  5. Display docker image IDs. You are probably interested in the most recently created image. </br>
  `$ docker images`
  6. Start the docker image and the Jupyter notebook server.</br>
   `$ docker run -i -t -p 8888:8888 <image_id> /bin/bash -c "/opt/conda/bin/conda install jupyter -y --quiet && /opt/conda/bin/jupyter notebook --notebook-dir=/mli-resources --ip='*' --port=8888 --no-browser"`
  7. List docker containers.</br>
  `$ docker ps`
  8. Copy the sample data into the Docker container. Refer to [GetData.md](data/GetData.md) to obtain datasets needed for notebooks.</br>
  `$ docker cp path/to/train.csv <container_id>:/mli-resources/data/train.csv`
  9. Navigate to port 8888 on your machine.

***

##### Manual

  Install:

  1. Anaconda Python 4.2.0 from the [Anaconda archives](https://repo.continuum.io/archive/).
  2. [Java](https://java.com/download).
  3. The latest stable [h2o](https://www.h2o.ai/download/) Python package.
  4. [Git](https://git-scm.com/downloads).
  5. [XGBoost](https://github.com/dmlc/xgboost) with Python bindings.
  6. [GraphViz](http://www.graphviz.org/).

  Anaconda Python, Java, Git, and GraphViz must be added to your system path.

  From a terminal:

  7. Clone the mli-resources repository with examples.</br>
  `$ git clone https://github.com/h2oai/mli-resources.git`
  8. `$ cd mli-resources`
  9. Copy the sample data into the mli-resources repo directory. Refer to [GetData.md](data/GetData.md) to obtain datasets needed for notebooks.</br>
  `$ cp path/to/train.csv ./data`
  9. Start the Jupyter notebook server.</br>
  `$ jupyter notebook`
  10. Navigate to the port Jupyter directs you to on your machine.

#### Additional Code Examples

The notebooks on in this repo have been revamped and refined many times. Other versions with different, and potentially interesting, details are available at these locations:

* [O'Reilly Media GitLab](https://content.oreilly.com/oriole/Interpretable-machine-learning-with-Python-XGBoost-and-H2O)
* [github.com/jphall663](https://github.com/jphall663/interpretable_machine_learning_with_python)

### Webinars/Videos

* [Interpretable Machine Learning Meetup - Washington DC](https://www.youtube.com/watch?v=3uLegw5HhYk)
* [Machine Learning Interpretability with Driverless AI](https://www.youtube.com/watch?v=3_gm00kBwEw)
* [Interpretability in conversation with Patrick Hall and Sameer Singh](http://blog.fastforwardlabs.com/2017/09/11/interpretability-webinar.html)
* O'Reilly Media Interactive Notebooks (Requires O'Reilly Safari Membership):
  * [Enhancing transparency in machine learning models with Python and XGBoost](https://www.safaribooksonline.com/oriole/enhancing-transparency-in-machine-learning-models-with-python-and-xgboost)
  * [Increase transparency and accountability in your machine learning project with Python](https://www.safaribooksonline.com/oriole/increase-transparency-and-accountability-in-your-machine-learning-project-with-python)
  * [Explain your predictive models to business stakeholders with LIME, Python, and H2O](https://www.safaribooksonline.com/oriole/explain-your-predictive-models-to-business-stakeholders-w-lime-python-h2o)
  * [Testing machine learning models for accuracy, trustworthiness, and stability with Python and H2O](https://www.safaribooksonline.com/oriole/testing-ml-models-for-accuracy-trustworthiness-stability-with-python-and-h2o)

### Booklets

* [An Introduction to Machine Learning Interpretability](http://www.oreilly.com/data/free/an-introduction-to-machine-learning-interpretability.csp)
* [Machine Learning Interpretability with H2O Driverless AI Booklet](http://docs.h2o.ai/driverless-ai/latest-stable/docs/booklets/MLIBooklet.pdf)

### Conference Presentations

* [Practical Techniques for Interpreting Machine Learning Models - 2018 FAT* Conference Tutorial](https://www.fatconference.org/static/tutorials/hall_interpretable18.pdf)
* [Driverless AI Hands-On Focused on Machine Learning Interpretability - H2O World 2017](http://video.h2o.ai/watch/9g8TrVXUfgYgKq4FReia7z)
* [Interpretable AI: Not Just For Regulators! - Strata NYC 2017](notes/strata_mli_sept_17.pdf)
* [Ideas on Interpreting Machine Learning - SlideShare](https://www.slideshare.net/0xdata/interpretable-machine-learning)

### Notes

* [Notes from Patrick Hall's data mining class at GWU](https://github.com/jphall663/GWU_data_mining/blob/master/10_model_interpretability/notes/instructor_notes.pdf)

### References

#### General

* [Towards A Rigorous Science of Interpretable Machine Learning](https://arxiv.org/pdf/1702.08608.pdf)
* [Ideas for Machine Learning Interpretability](https://www.oreilly.com/ideas/ideas-on-interpreting-machine-learning)
* [Fairness, Accountability, and Transparency in Machine Learning (FAT/ML) Scholarship](https://www.fatml.org/resources/relevant-scholarship)

#### Techniques

* **Partial Dependence**: [*Elements of Statistical Learning*](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12.pdf)
* **LIME**: [“Why Should I Trust You?” Explaining the Predictions of Any Classifier](http://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf)
* **LOCO**: [Distribution-Free Predictive Inference for Regression](http://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf)
* **ICE**: [Peeking inside the black box: Visualizing statistical learning with plots of individual conditional expectation](https://arxiv.org/pdf/1309.6392.pdf)
* **Surrogate Models**
  * [Extracting tree structured representations of trained networks](https://papers.nips.cc/paper/1152-extracting-tree-structured-representations-of-trained-networks.pdf)
  * [Interpreting Blackbox Models via Model Extraction](https://arxiv.org/pdf/1705.08504.pdf)
* **Shapely Explanations**: [A Unified Approach to Interpreting Model Predictions](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)
* **Anchors**: [Anchors: High-Precision Model-Agnostic Explanations](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf)
