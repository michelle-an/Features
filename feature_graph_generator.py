import argparse
import os
import sys

import numpy as np
import pandas
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor


def get_args():
    # create the parser
    parser = argparse.ArgumentParser(
        description="Feature Graph Generator: A tool to explore feature quality by producing pretty graphs from the "
        "intermediate csv file of calculated features.",
    )

    # adding parser arguments
    parser.add_argument(
        "-i",
        action="store",
        type=str,
        dest="input",
        help="the path to the input file",
    )
    parser.add_argument(
        "-o",
        action="store",
        type=str,
        dest="output",
        help="optional flag to set the path to the output directory",
        default=os.getcwd(),
    )
    parser.add_argument(
        "-c",
        action="append",
        type=str,
        dest="class_list",
        help="optional flag for selecting only specific classes to be graphed, may enter multiple by calling this "
        "flag multiple times",
        default=None,
    )
    parser.add_argument(
        "-s",
        action="store_true",
        dest="show",
        help='optional flag to set the "show graphs" switch to True',
        default=False,
    )
    parser.add_argument(
        "-v",
        action="store_true",
        dest="verbose",
        help='optional flag to set the "verbose" switch to True',
        default=False,
    )
    parser.add_argument(
        "--skip",
        action="append",
        dest="skip",
        choices=["tval", "rf", "corr", "joy"],
        help="optional flag to skip generating certain graphs. May be called multiple times",
        default=[],
    )

    # executing the parse_args command
    args = parser.parse_args()

    # getting args
    input_file = args.input
    output_file = args.output
    class_list = args.class_list
    show = args.show
    verbose = args.verbose
    skip = args.skip

    return input_file, output_file, class_list, show, verbose, skip


def cat_cont_correlation_ratio(categories, values):
    """
    Simple function to determine the correlation ratio between a list
    of categorical values and a list of continuous values.
    Code provided by Julien.
    """
    f_cat, _ = pandas.factorize(categories)
    cat_num = np.max(f_cat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[np.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def correlation_bar_plots(df, outfile, class_list, show):

    # get a list of the name of each feature if the first value in the dataframe in the target column is a number
    features_list = [x for x in df.columns.to_list() if type(df[x][0]) == np.float64 or type(df[x][0]) == np.int64]
    # sort it alphabetically ignoring case
    features_list = sorted(features_list, key=str.lower)

    # set up an empty array for correlation values. Rows = classes, Columns = features
    correlation_array = np.zeros((len(class_list), len(features_list)))

    for i in range(len(class_list)):
        classification = class_list[i]

        # figure out which rows are the correct classification
        is_target_class = [
            True if x == classification else False for x in df["classification"]
        ]

        # loop through each feature from the list
        for ii in range(len(features_list)):
            feature = features_list[ii]
            # run the correlation function between it and the classification column
            correlation_array[i][ii] = round(
                cat_cont_correlation_ratio(is_target_class, df[feature]), 5
            )

    correlation_plot = go.Figure(
            data=go.Heatmap(z=correlation_array,
                            x=features_list,
                            y=class_list,
                            hoverongaps=False, colorscale='Plasma'),
    )

    correlation_plot.update_layout(
        xaxis_title="Features",
        yaxis_title="Phage Protein Class",
        title_text=f"Correlation Ratios",
        font=dict(size=12),
    )
    correlation_plot.update_xaxes(tickangle=45, tickfont=dict(size=12))

    correlation_plot.write_html(
        file=f"{outfile}/correlation heatplot.html",
        include_plotlyjs=True,
    )

    if show:
        correlation_plot.show()

    return


def t_value_bar_plots(df, outfile, class_list, show):

    # get a list of the name of each feature if the first value in the dataframe in the target column is a number
    features_list = [x for x in df.columns.to_list() if type(df[x][0]) == np.float64 or type(df[x][0]) == np.int64]
    # sort it alphabetically ignoring case
    features_list = sorted(features_list, key=str.lower)

    # set up an empty array for correlation values. Rows = classes, Columns = features
    tval_array = np.zeros((len(class_list), len(features_list)))

    for i in range(len(class_list)):
        classification = class_list[i]

        # figure out which rows are the correct classification
        is_target_class = [
            True if x == classification else False for x in df["classification"]
        ]

        # loop through each column in the data frame and check if the first row value is a number of some kind
        for ii in range(len(features_list)):
            feature = features_list[ii]
            if type(df[feature][0]) == np.float64 or type(df[feature][0]) == np.int64:
                # if it is, get the t-value. Following code was provided by Julien.
                predictor = statsmodels.api.add_constant(df[feature].to_numpy())

                logistic_regression_model = statsmodels.api.Logit(
                    is_target_class, predictor
                )
                logistic_regression_fitted = logistic_regression_model.fit(disp=False)

                t_value = round(logistic_regression_fitted.tvalues[1], 4)
                tval_array[i][ii] = abs(t_value)

    t_val_plot = go.Figure(
        data=go.Heatmap(z=tval_array,
                        x=features_list,
                        y=class_list,
                        hoverongaps=False, colorscale='Plasma'),
    )

    t_val_plot.update_layout(
        xaxis_title="Features",
        yaxis_title="Phage Protein Class",
        title_text=f"T-Values",
        font=dict(size=12),
    )
    t_val_plot.update_xaxes(tickangle=45, tickfont=dict(size=12))


    t_val_plot.write_html(
        file=f"{outfile}/t-value heatplot.html",
        include_plotlyjs=True,
    )

    if show:
        t_val_plot.show()

    return


def t_value_bar_plots_old(df, outfile, class_list, show):
    for classification in class_list:

        # set up empty dictionary
        t_values = {}

        # figure out which rows are the correct classification
        is_target_class = [
            True if x == classification else False for x in df["classification"]
        ]

        # loop through each column in the data frame and check if the first row value is a number of some kind
        for feature in df.columns.to_list():
            if type(df[feature][0]) == np.float64 or type(df[feature][0]) == np.int64:
                # if it is, get the t-value. Following code was provided by Julien.
                predictor = statsmodels.api.add_constant(df[feature].to_numpy())

                logistic_regression_model = statsmodels.api.Logit(
                    is_target_class, predictor
                )
                logistic_regression_fitted = logistic_regression_model.fit(disp=False)

                t_value = round(logistic_regression_fitted.tvalues[1], 4)
                t_values[feature] = abs(t_value)

        # set x and y values for plotting
        x_axis = list(t_values.keys())
        y_axis = list(t_values.values())

        # sort alphabetically using zip sort and then return data to list format
        x_axis, y_axis = zip(*sorted(zip(x_axis, y_axis)))
        x_axis, y_axis = list(x_axis), list(y_axis)

        # set up t-value plot with layout options
        t_val_plot = go.Figure(
            [
                go.Bar(
                    x=x_axis, y=y_axis, marker={"color": y_axis, "colorscale": "dense"}
                )
            ]
        )
        t_val_plot.update_layout(
            xaxis_title="Features",
            yaxis_title="|t-value|",
            title_text=f"t-values - {classification}",
            font=dict(size=12),
        )
        t_val_plot.update_xaxes(tickangle=45, tickfont=dict(size=10))

        t_val_plot.write_html(
            file=f"{outfile}/t-value {classification}.html",
            include_plotlyjs=True,
        )

        if show:
            t_val_plot.show()
    return


def feature_importance_bar_plot(df, outfile, class_list, show):
    for classification in class_list:

        # figure out which rows are the correct classification, convert into data frame
        is_target_class = [
            True if x == classification else False for x in df["classification"]
        ]
        is_target_class = pandas.DataFrame(is_target_class, columns=["classification"])

        # setup second data frame that drops all non-testable columns
        df_rf = df
        for feature in df_rf.columns.to_list():
            if (
                type(df_rf[feature][0]) != np.float64
                and type(df_rf[feature][0]) != np.int64
            ):
                df_rf = df_rf.drop(feature, axis=1)

        # convert to numpy and flatten the is_target_class array to make RandomForestRegressor() happy
        is_target_class_array = np.ravel(is_target_class.to_numpy(), order="C")
        df_rf_array = df_rf.to_numpy()

        # get feature importance for each column in the data frame
        rf = RandomForestRegressor()
        rf.fit(df_rf_array, is_target_class_array)
        feature_importance = rf.feature_importances_

        # set x and y values for plotting
        x_axis = list(df_rf.columns)
        y_axis = list(feature_importance)

        # sort numerically using zip sort and then return data to list format
        y_axis, x_axis = zip(*sorted(zip(y_axis, x_axis)))
        x_axis, y_axis = list(x_axis), list(y_axis)

        # set up correlation plot with layout options
        rf_plot = go.Figure(
            [
                go.Bar(
                    x=x_axis, y=y_axis, marker={"color": y_axis, "colorscale": "Blugrn"}
                )
            ]
        )
        rf_plot.update_layout(
            xaxis_title="Features",
            yaxis_title="RF Feature Importance Metric",
            title_text=f"Random Forest Feature Importance - {classification}",
            font=dict(size=12),
        )
        rf_plot.update_xaxes(tickangle=45, tickfont=dict(size=10))

        rf_plot.write_html(
            file=f"{outfile}/RF feature importance {classification}.html",
            include_plotlyjs=True,
        )
        if show:
            rf_plot.show()
    return


def joy_plot(df, outfile, class_list, show):
    for feature in df.columns.to_list():
        if type(df[feature][0]) == np.float64 or type(df[feature][0]) == np.int64:
            df_to_plot = df[["classification", feature]]
            df_to_plot = df_to_plot[df["classification"].isin(class_list)]
            joy_violin = px.violin(
                df_to_plot,
                x=feature,
                color="classification",
                violinmode="overlay",
                points=False,
                orientation="h",
                color_discrete_sequence=px.colors.qualitative.Pastel1,
            )
            joy_violin.update_layout(height=400)
            joy_violin.update_traces(width=0.9, points=False)
            joy_violin.update_yaxes(range=[0, 1])

            joy_violin.update_layout(title=f"Joy Plot of {feature}")
            joy_violin.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)

            joy_violin.write_html(
                file=f"{outfile}/joy plot {feature}.html",
                include_plotlyjs="cdn",
            )
            if show:
                joy_violin.show()
    return


def main():
    # call argparse function
    infile, outfile, classes, show, verbose, skip = get_args()

    # if output directory doesn't exist, create it
    if not os.path.exists(outfile):
        if verbose:
            print("Creating new output directory")
        os.makedirs(outfile)

    # open dataframe from csv
    if verbose:
        print(f"Reading in data from {infile}")
    df = pandas.read_csv(infile, engine='python').dropna(axis=0).reset_index()
    if "index" in df.columns.to_list():
        df.drop("index", axis=1, inplace=True)

    # if no classes assigned, set the classes to ALL classes in the dataframe
    if classes is None:
        if verbose:
            print("No class list provided. Using ALL classes")
        classes = list(set(df["classification"].values))

    # generate joy plots for each feature
    if "joy" not in skip:
        if verbose:
            print("Generating joy plots")
        joy_plot(df, outfile, classes, show)
    else:
        if verbose:
            print("Skipping joy plots")

    # generate stand-alone correlation bar plots
    if "corr" not in skip:
        if verbose:
            print("Generating correlation plots")
        correlation_bar_plots(df=df, outfile=outfile, class_list=classes, show=show)
    else:
        if verbose:
            print("Skipping correlation plots")

    # generate stand-alone t-value bar plots
    if "tval" not in skip:
        if verbose:
            print("Generating t-value plots")
        t_value_bar_plots(df=df, outfile=outfile, class_list=classes, show=show)
    else:
        if verbose:
            print("Skipping t-value plots")

    # generate RF feature importance plots
    if "rf" not in skip:
        if verbose:
            print("Generating feature importance plots. This may take some time...")
        feature_importance_bar_plot(
            df=df, outfile=outfile, class_list=classes, show=show
        )
    else:
        if verbose:
            print("Skipping feature importance plots")

    if verbose:
        print("Success!")


if __name__ == "__main__":
    main()

# TODO: make correlation graphs, RF graphs, and t-value graphs shorter height
