# Copyright 2018 Patrick Hall (phall@h2o.ai), Lingyao Meng (danielle@h2o.ai) 
# and the H2O.ai team.

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import string
import numpy as np
import pandas as pd
import h2o
from numpy.random import uniform

class DataMakerAndGetter(object):

    """ This class makes randomly generated data sets for testing the H2O.ai
    MLI module and contains other data utility functions. Generated data sets
    currently contain 15 columns of inputs - 3 of which are categorical - and
    one target variable. Randomly generated inputs are not correlated with
    one another or the target. Users may select to the number of rows in the
    generated data set, the type of target, and whether to save the generated
    data.

    :ivar seed: Random seed for enhanced reproducibility, default 12345.
    :ivar nrows: Number of rows of the  generated data set, default 100000.
    :ivar target: Can be 'numeric' or 'binary', default 'binary'.
    :ivar save: Save or not save the file, default not save.
    :ivar noise: Adds noise to signal generating function, default False.

    """

    # constructor
    def __init__(self, seed=None, nrows=None, target=None, save=None,
                 one_function=None, noise=None):

        # init properties
        self.__ncols = 15
        self.__ncats = 3

        # set ivar defaults
        if seed is None:
            self.seed = 12345
        else:
            self.seed = seed

        if nrows is None:
            self.nrows = 100000
        else:
            self.nrows = nrows

        if target is None:
            self.target = 'binary'
        else:
            self.target = target

        if save is None:
            self.save = False
        else:
            self.save = save

        if one_function is None:
            self.one_function = False
        else:
            self.one_function = one_function

        if noise is None:
            self.noise = False
        else:
            self.noise = noise

    @property
    def ncols(self):

        """ Number of variables of the generated data set, fixed at 15. """

        return self.__ncols

    @property
    def ncats(self):

        """ Number of categorical variables in the generated data set,
        fixed at 3. """

        return self.__ncats

    def make_random(self):

        """ This function makes a random data set with user defined number of
        rows and type of target variable and can save the generated set as a
        CSV file. The default setting is 100000*15 variables, in which 3
        variables are categorical, plus one target column.

        :return: Returns a random H2OFrame with user defined number of rows
        and target variable type.

        """

        # set random seed
        np.random.seed(self.seed)

        # create numeric frame
        n_num_cols = self.ncols - self.ncats
        num_col_names = ['num' + str(i + 1) for i in range(0, n_num_cols)]
        num_cols = pd.DataFrame(np.random.randn(self.nrows, n_num_cols),
                                columns=num_col_names)

        # make categorical frame
        cat_col_names = map(lambda j: 'cat' + str(j + 1),
                            range(0, self.ncats))
        text_draw = [(letter * 8) for letter in string.ascii_uppercase[:7]]
        cat_cols = pd.DataFrame(np.random.choice(text_draw, (self.nrows,
                                                             self.ncats)),
                                columns=cat_col_names)

        # make target frame
        if self.target == 'binary':
            target_ = pd.DataFrame(np.random.choice([0, 1], size=self.nrows,
                                                    p=[0.5, 0.5]),
                                   columns=['target'])
        else:
            target_ = pd.DataFrame(np.random.randint(100, size=self.nrows),
                                   columns=['target'])

        # column bind all frames together
        frame = pd.concat([num_cols, cat_cols, target_], axis=1)

        # add row_id
        frame['row_id'] = frame.index

        # conditionally save
        if self.save:
            frame.to_csv('random.csv', index=False)

        return h2o.H2OFrame(frame)

    def make_random_with_signal(self):

        """ This function transforms the dataset generated by make_random. The
        number of total variables are fixed at 15, with 3 of them being
        categorical. 9 of the numeric variable, num1 to num9, were used to transform
        the target. 3 of the numeric columns are totally random.
        The generated set can also be saved.

        :return: Returns an H2OFrame with signal of user defined
        number of rows and type of target.

        """

        # local constants and settings
        cached_save = self.save
        self.save = False

        # make totally random frame
        frame = self.make_random().as_data_frame()

        # function 1, on all rows
        frame.loc[::1, 'target'] = np.abs(frame.loc[::1, 'num8']) *\
            np.square(frame.loc[::1, 'num9']) + frame.loc[::1, 'num4'] *\
            frame.loc[::1, 'num1']

        frame.loc[::1, 'function'] = 1

        # function 2, on mod 2 rows
        if not self.one_function:
            frame.loc[::2, 'target'] = np.sin(frame.loc[::2, 'num6']) -\
                np.sqrt(np.abs(frame.loc[::2, 'num7'])) - frame.loc[::2, 'num2'] *\
                frame.loc[::2, 'num3']

            frame.loc[::2, 'function'] = 2

        cut = frame.loc[:, 'target'].mean()

        if self.noise:
            frame.loc[::5, 'target'] = 1
            frame.loc[::7, 'target'] = 0

        # set the cut off point for binary target
        if self.target == 'binary':
            frame.loc[frame['target'] >= cut, 'target'] = 1
            frame.loc[frame['target'] < cut, 'target'] = 0

        # conditionally save
        if cached_save:
            frame.to_csv('random_with_signal.csv', index=False)
            self.save = cached_save

        return h2o.H2OFrame(frame)

    @classmethod
    def get_percentile_dict(cls, y, id_, frame):

        """ Returns the percentiles of a column, yhat, as the indices based on
        another column id_. Expects H2OFrame.

        :param y: Column in which to find percentiles.
        :param id_: Id column that stores indices for percentiles of yhat.
        :param frame: H2OFrame containing y and id_.

        :returns: Dictionary of percentile values and index column values.

        """

        # create a copy of frame and sort it by yhat
        sort_df = frame.as_data_frame()
        sort_df.sort_values(y, inplace=True)
        sort_df.reset_index(inplace=True)

        # find top and bottom percentiles
        percentiles_dict = {0: sort_df.loc[0, id_],
                            99: sort_df.loc[sort_df.shape[0] - 1, id_]}

        # find 10th-90th percentiles
        inc = sort_df.shape[0] // 10
        for i in range(1, 10):
            percentiles_dict[i * 10] = sort_df.loc[i * inc, id_]

        return percentiles_dict
