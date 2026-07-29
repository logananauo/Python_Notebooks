###################################################
### Scikit Learn Libraries for Machine Learning ###
###################################################


### datasets
from sklearn.datasets import(
    load_iris,
    load_wine,
    load_digits,
    load_breast_cancer,
    make_classification,
    make_regression,
    make_blobs
)


### model selection
from sklearn.model_selection import(
    train_test_split,
    KFold,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score
)


### preprocessing
from sklearn.preprocessing import(
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
    PolynomialFeatures
)


### pipelines
from sklearn.pipeline import(
    Pipeline,
    make_pipeline
)


### metrics
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


### linear models
from sklearn.linear_model import(
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso,
    ElasticNet,
    SGDClassifier,
    SGDRegressor
)


### tree models
from sklearn.tree import(
    DecisionTreeClassifier,
    DecisionTreeRegressor
)


### ensemble methods
from sklearn.ensemble import(
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    BaggingClassifier,
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor
)


### support vector machines
from sklearn.svm import(
    SVC,
    SVR,
    LinearSVC,
    LinearSVR
)


### neighbors
from sklearn.neighbors import(
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors
)


### naive bayes
from sklearn.naive_bayes import(
    GaussianNB,
    MultinomialNB,
    BernoulliNB
)


### neural networks
from sklearn.neural_network import(
    MLPClassifier,
    MLPRegressor
)


### clustering
from sklearn.cluster import(
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    Birch,
    MeanShift,
    SpectralClustering
)


### dimensionality reduction
from sklearn.decomposition import(
    PCA,
    TruncatedSVD,
    FastICA,
    NMF
)


### feature selection
from sklearn.feature_selection import(
    SelectKBest,
    SelectFromModel,
    RFE,
    RFECV
)

