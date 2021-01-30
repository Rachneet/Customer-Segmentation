import plotly_express as px
import plotly.graph_objects as go
import plotly


def update_axis(fig, axis, dtick=None, title=None, fontsize=15):
    """
    :param fig: plotly figure object
    :param axis: axis to modify 'x' or 'y'
    :param dtick: integer value for tick spacing
    :param title: axis title
    :param fontsize: label font size
    :return: None
    """
    if axis == 'x':
        fig.update_xaxes(
            showline=True,
            ticks='outside',
            mirror=True,
            # tickfont=dict(family="times new roman", size=15, color='black'),
            linecolor='black',
            #             tick0=0,
            dtick=dtick,
            linewidth=1,
            title=dict(
                font=dict(
                    # family="sans-serif",
                    size=fontsize,
                    color="black"
                ),
                text=title
            )
        )

    elif axis == 'y':
        fig.update_yaxes(
            showline=True,
            ticks='outside',
            mirror=True,
            # tickfont=dict(family="times new roman", size=15, color='black'),
            linecolor='black',
            #             tick0=0,
            dtick=dtick,
            linewidth=1,
            title=dict(
                font=dict(
                    # family="sans-serif",
                    size=fontsize,
                    color="black"
                ),
                text=title
            )
        )


def update_layout(fig, title=None, width=None, height=None, plotbg='rgba(0,0,0,0)', showlegend=False):
    """
    :param fig: plotly figure object
    :param title: title of the plot
    :param width: plot width
    :param height: plot height
    :param plotbg: plot background color
    :param showlegend: boolean value to show legend or hide it
    :return:
    """
    fig.update_layout(
        title_text=title,
        # margin=dict(b=260, l=0, r=150, t=20), # for small boxes
        #         margin=dict(b=350, l=0, r=200, t=20),  # normal
        title_x=0.50,
        # title_y=0.90,
        paper_bgcolor='white',
        plot_bgcolor=plotbg,
        showlegend=showlegend,
        legend=dict(
            bordercolor='black',
            borderwidth=1,
            bgcolor='rgba(0,0,0,0)',
            orientation='h',
            itemsizing='constant',
            x=0,
            y=1.2,
            font=dict(
                # family="sans-serif",
                size=10,
                color="black"
            ),
            traceorder='normal'
        ),
        width=width,
        height=height
    )


def save_fig(fig, figname, width=None, height=None):
    """
    :param fig: plotly figure object
    :param figname: name of saved figure
    :param width: figure width
    :param height: figure height
    :return:
    """
    plotly.offline.plot(figure_or_data=fig, image_width=width, image_height=height, filename=figname, image='svg')


def draw_comparison_chart(x, y1, y2, save=False):
    """
    :param x: array values for the x axis
    :param y1: array values for the y axis
    :param y2: array values for the y axis
    :param save: boolean value stating whether to save the plot or not
    :return: None
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x,
        y=y1,
        name="General population",
        textfont=dict(color='black', size=10),
        textposition='auto',
        marker_color='#23aaff',  # blue
        marker_line_color='#1f77b4',
    ))

    fig.add_trace(go.Bar(
        x=x,
        y=y2,
        name="Customer population",
        textfont=dict(color='black', size=10),
        textposition='auto',
        marker_color='#66c56c',  # green
        marker_line_color='#2ca02c',

    ))

    fig.update_yaxes(automargin=True,
                     showline=True,
                     ticks='inside',
                     mirror=True,
                     tickfont=dict(family="times new roman", size=15, color='black'),
                     linecolor='black',
                     linewidth=1,
                     title=dict(
                         font=dict(
                             # family="sans-serif",
                             size=15,
                             color="black"
                         ),
                         text='Percent population')
                     )
    fig.update_xaxes(automargin=True, side='bottom',
                     showline=True,
                     ticks='outside',
                     mirror=True,
                     tickfont=dict(family="times new roman", size=15, color='black'),
                     linecolor='black',
                     linewidth=1,
                     dtick=1,
                     title=dict(
                         font=dict(
                             # family="sans-serif",
                             size=15,
                             color="black"
                         ),
                         text='Clusters'
                     )
                     )

    # blue #23aaff, red apple #ff6555, moss green #66c56c, mustard yellow #f4b247
    fig.update_traces(marker_line_width=1, opacity=0.8)

    fig.update_layout(
        title_text='<b>Comparison of general and customer population <br>amongst the clusters.',
        # margin=dict(b=260, l=0, r=150, t=20), # for small boxes
        #         margin=dict(b=350, l=0, r=200, t=20),  # normal
        title_x=0.50,
        title_y=0.90,
        yaxis={"mirror": "all"},
        paper_bgcolor='white',
        plot_bgcolor='rgba(0,0,0,0)',
        bargap=0.2,
        barmode='group',
        bargroupgap=0.1,
        legend=dict(
            bordercolor='black',
            borderwidth=1,
            bgcolor='rgba(0,0,0,0)',
            orientation='h',
            itemsizing='constant',
            # x=0.4,  # for hw comp
            x=0.3,  # for model comp
            y=0.9,
            font=dict(
                # family="sans-serif",
                size=10,
                color="black"
            ),
            traceorder='normal'
        )
    )
    fig.show()
    if save:
        plotly.offline.plot(figure_or_data=fig, filename='cluster_pop.html', image='svg')
