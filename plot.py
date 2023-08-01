import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
    
def low_pass_filter(x, filter_factor=0.9):
    y_list = []
    y = x[0]
    
    for i in range(len(x)):
        y = y * filter_factor + x[i] * (1 - filter_factor)
        y_list.append(y)
        
    return y_list

def configure_axis(ax, yscale='', title='', xlabel='', ylabel='', x_start=1, x_end=None, y_start=1, y_end=None, title_pad=None, 
                   font_title=56, font_x_label=56, font_y_label=56, font_tick=48, grid_linewidth=4, outline_width=4):
    
    if yscale != '':
        ax.set_yscale(yscale)
        
    if title != '':
        if title_pad != None:
            ax.set_title(title, fontsize=font_title, pad=title_pad)
        else:
            ax.set_title(title, fontsize=font_title)
    
    if xlabel != '':
        ax.set_xlabel(xlabel, fontsize=font_x_label)
        
    if ylabel != '':
        ax.set_ylabel(ylabel, fontsize=font_y_label)
        
    if x_end != None:
        ax.set_xlim([x_start, x_end])
        
    if y_end != None:
        ax.set_ylim([y_start, y_end])
    
    ax.tick_params(axis='x', which='both', labelsize=font_tick)
    ax.tick_params(axis='y', which='both', labelsize=font_tick)
    
    if grid_linewidth != 0:
        ax.grid(True, which="both", linewidth=grid_linewidth)
    
    if outline_width != 0:
        [i.set_linewidth(outline_width) for i in ax.spines.values()]
    
def save_and_show_plt(plt, save_to_file):
    if save_to_file != '':
        plt.savefig(f'{save_to_file}.png')
    
    plt.show()

def plot_multiple_agents(agents_data, filter_factor=0.9, show_non_filtered=True, transparency=0.4, show_loss=False, loss_skip=20, 
                         use_best_agents=False, plot_eval=False, save_to_file='', 
                         size_x=48, size_y=12, dpi=200, linewidth=4, linewidth_filtered=6, markersize=24, 
                         font_title=56, font_x_label=56, font_y_label=56, font_tick=48, grid_linewidth=4, outline_width=4):
    
    if plot_eval:
        histories = [i[2] for i in agents_data]
        run_lists = [i[0] for i in histories]
        score_mean_lists = [i[1] for i in histories]
        episode_lengths_mean_lists = [i[2] for i in histories]
        
    if use_best_agents:
        agents = [i[1] for i in agents_data]
        
    else:
        agents = [i[0] for i in agents_data]

    if show_loss:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
    fig.set_size_inches(size_x, size_y)
    fig.set_dpi(dpi)
    
    max_ep = 0
    for i in range(len(agents)):
        if len(agents[i].score_hist) > max_ep:
            max_ep = len(agents[i].score_hist)
    
    for i in range(len(agents)):
        x = np.arange(1, len(agents[i].score_hist) + 1)
        
        if show_non_filtered:
            ax1.plot(x, -np.array(agents[i].score_hist), linewidth=linewidth, alpha=transparency, color=f'C{i}')
            
        ax1.plot(x, -np.array(low_pass_filter(agents[i].score_hist, filter_factor)), 
                 linewidth=linewidth_filtered, color=f'C{i}')
        
        if plot_eval:
            ax1.plot(run_lists[i], -np.array(score_mean_lists[i]), 'o', color=f'C{i}', markersize=markersize)
        
        configure_axis(ax1, yscale='log', title='Accumulated Cost', xlabel='Episode', x_end=max_ep, 
                       font_title=font_title, font_x_label=font_x_label, font_y_label=font_y_label, 
                       font_tick=font_tick, grid_linewidth=grid_linewidth, outline_width=outline_width)

        if show_non_filtered:
            ax2.plot(x, agents[i].t_max_hist, linewidth=linewidth, alpha=transparency, color=f'C{i}')
            
        ax2.plot(x, low_pass_filter(agents[i].t_max_hist, filter_factor), linewidth=linewidth_filtered, color=f'C{i}')
        
        if plot_eval:
            ax2.plot(run_lists[i], episode_lengths_mean_lists[i], 'o', color=f'C{i}', markersize=markersize)
            
        configure_axis(ax2, yscale='log', title='Episode Length (s)', xlabel='Episode', x_end=max_ep, 
                       font_title=font_title, font_x_label=font_x_label, font_y_label=font_y_label, 
                       font_tick=font_tick, grid_linewidth=grid_linewidth, outline_width=outline_width)
        
        if show_loss:
            loss = agents[i].loss_hist.copy()
            x = np.arange(loss_skip + 1, len(loss) + 1)
            loss = loss[loss_skip:len(loss)]
            ax3.plot(x, loss, linewidth=linewidth, color=f'C{i}')
            
            configure_axis(ax3, yscale='log', title='Loss', xlabel='Episode', x_end=max_ep, 
                           font_title=font_title, font_x_label=font_x_label, font_y_label=font_y_label, 
                           font_tick=font_tick, grid_linewidth=grid_linewidth, outline_width=outline_width)
    
    fig.tight_layout()
    
    save_and_show_plt(plt, save_to_file)

def plot_compare_trainings(agent_DDQN, agent_TD3, filter_factor=0.9, show_non_filtered=True, transparency=0.4, show_loss=False, loss_skip=20, 
                           use_best_agents=False, save_to_file='', 
                           size_x=48, size_y=12, dpi=200, linewidth=4, linewidth_filtered=6, markersize=24, font_legend=48, 
                           font_title=48, font_x_label=48, font_y_label=48, font_tick=44, grid_linewidth=4, outline_width=4):
    
    agents = [agent_DDQN, agent_TD3]
    labels = ['DDQN', 'TD3']
    
    if use_best_agents:
        agents = [i[1] for i in agents]
        
    else:
        agents = [i[0] for i in agents]
        
    if show_loss:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        
    fig.set_size_inches(size_x, size_y)
    fig.set_dpi(dpi)
    
    max_ep = 0
    for i in range(len(agents)):
        if len(agents[i].score_hist) > max_ep:
            max_ep = len(agents[i].score_hist)
    
    for i in range(len(agents)):
        x = np.arange(1, len(agents[i].score_hist) + 1)
        
        if show_non_filtered:
            ax1.plot(x, -np.array(agents[i].score_hist), linewidth=linewidth, alpha=transparency, color=f'C{i}')
            
        ax1.plot(x, -np.array(low_pass_filter(agents[i].score_hist, filter_factor)), linewidth=linewidth_filtered, color=f'C{i}')
        
        configure_axis(ax1, yscale='log', title='', xlabel='Episode', ylabel='Accumulated Cost', x_end=max_ep, 
                       font_title=font_title, font_x_label=font_x_label, font_y_label=font_y_label, 
                       font_tick=font_tick, grid_linewidth=grid_linewidth, outline_width=outline_width)

        if show_non_filtered:
            ax2.plot(x, agents[i].t_max_hist, linewidth=linewidth, alpha=transparency, color=f'C{i}', label=labels[i])
            ax2.plot(x, agents[i].t_max_hist, linewidth=linewidth, alpha=transparency, color=f'C{i}')
            
        ax2.plot(x, low_pass_filter(agents[i].t_max_hist, filter_factor), linewidth=linewidth_filtered, 
                 color=f'C{i}', label=f'{labels[i]} (filtered)')

        ax2.legend(loc='lower right', fontsize=font_legend)
        
        configure_axis(ax2, yscale='log', title='', ylabel='Episode Length (s)', xlabel='Episode', x_end=max_ep, 
                       font_title=font_title, font_x_label=font_x_label, font_y_label=font_y_label, 
                       font_tick=font_tick, grid_linewidth=grid_linewidth, outline_width=outline_width)
        
        if not show_loss:
            ax2.set_xlabel('Episode', fontsize=font_x_label);
        
        if show_loss:
            loss = agents[i].loss_hist.copy()
            x = np.arange(loss_skip + 1, len(loss) + 1)
            loss = loss[loss_skip:len(loss)]
            
            ax3.plot(x, loss, linewidth=linewidth, color=f'C{i}')
            
            configure_axis(ax3, yscale='log', title='Loss', xlabel='Episode', x_end=max_ep, 
                           font_title=font_title, font_x_label=font_x_label, font_y_label=font_y_label, 
                           font_tick=font_tick, grid_linewidth=grid_linewidth, outline_width=outline_width)
    
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.26)
    
    save_and_show_plt(plt, save_to_file)
    
def plot_both_tests(model_DDQN, model_TD3, save_to_file='', 
                    y_start_theta=None, y_end_theta=None, y_start_x=None, y_end_x=None, title_pad=35, 
                    size_x=48, size_y=36, dpi=200, linewidth=8, font_title=96, font_x_label=96, font_y_label=96, 
                    font_tick=84, grid_linewidth=8, outline_width=8):
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.set_size_inches(size_x, size_y)
    fig.set_dpi(dpi)
    
    plot_tests(model_DDQN, ax1, ax3, method='DDQN', title_pad=title_pad, 
               y_start_theta=y_start_theta, y_end_theta=y_end_theta, y_start_x=y_start_x, y_end_x=y_end_x, 
               linewidth=linewidth, font_title=font_title, font_x_label=font_x_label, font_y_label=font_y_label, 
               font_tick=font_tick, grid_linewidth=grid_linewidth, outline_width=outline_width)
    
    plot_tests(model_TD3, ax2, ax4, method='TD3', title_pad=title_pad, 
               y_start_theta=y_start_theta, y_end_theta=y_end_theta, y_start_x=y_start_x, y_end_x=y_end_x, 
               linewidth=linewidth, font_title=font_title, font_x_label=font_x_label, font_y_label=font_y_label, 
               font_tick=font_tick, grid_linewidth=grid_linewidth, outline_width=outline_width)
    
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.36, wspace=0.26)
    
    save_and_show_plt(plt, save_to_file)
    
def plot_tests(model, ax1, ax2, method='', 
               y_start_theta=None, y_end_theta=None, y_start_x=None, y_end_x=None, title_pad=35, 
               linewidth=8, font_title=96, font_x_label=96, font_y_label=96, font_tick=88, grid_linewidth=8, outline_width=8):
    
    max_t = 0.0
    
    for i in range(len(model)):
        if max(model[i].get_t()) > max_t:
            max_t = max(model[i].get_t())

        ax1.plot(model[i].get_t(), model[i].get_p_180(), linewidth=linewidth)
        
        configure_axis(ax1, title=f'{method} Inclination Control', xlabel=r'$t$ (s)', ylabel=r'$\theta$ (°)', 
                       x_start=0, x_end=max_t, y_start=y_start_theta, y_end=y_end_theta, title_pad=title_pad, 
                       font_title=font_title, font_x_label=font_x_label, font_y_label=font_y_label, 
                       font_tick=font_tick, grid_linewidth=grid_linewidth, outline_width=outline_width)

        ax2.plot(model[i].get_t(), model[i].get_x() * 100, linewidth=linewidth)
        
        configure_axis(ax2, title=f'{method} Position Control', xlabel=r'$t$ (s)', ylabel=r'$x$ (cm)', 
                       x_start=0, x_end=max_t, y_start=y_start_x, y_end=y_end_x, title_pad=title_pad, 
                       font_title=font_title, font_x_label=font_x_label, font_y_label=font_y_label, 
                       font_tick=font_tick, grid_linewidth=grid_linewidth, outline_width=outline_width)
        
        ax1.tick_params(axis='both', which='major', pad=15)
        ax2.tick_params(axis='both', which='major', pad=15)
   
def plot_policy_heatmap(agent, p_lim=14.0, x_lim=0.5, points=51, ticks_x=5, ticks_y=5, save_to_file='', title='', title_pad=25, 
                        size_x=32, size_y=12, dpi=200, font_title=56, font_x_label=56, font_y_label=56, font_tick=48):
    
    p = np.linspace(-p_lim, p_lim, points)
    x = np.linspace(0, x_lim, points)
    a = np.zeros([len(p), len(x)])
    
    for j in range(len(x)):
        for i in range(len(p)):
            a[i, j] = agent.best_action_v(np.array([x[j], 0, p[i] * math.pi / 180.0, 0]))

    fig = plt.figure(num=None, dpi=dpi, facecolor='w', edgecolor='k') 
    fig.set_size_inches(size_x, size_y)
    
    ax = sns.heatmap(a, cbar_kws={'label': r'Action $v$' + ' (V)'}, center=0, cmap="coolwarm")
    ax.collections[0].colorbar.set_ticks(np.linspace(-12, 12, ticks_y))
    
    configure_axis(ax, xlabel=r'Position $x$' + ' (cm)', ylabel=r'Inclination $\theta$' + ' (°)', 
                   font_title=font_title, font_x_label=font_x_label, font_y_label=font_y_label, 
                   font_tick=font_tick, grid_linewidth=0, outline_width=0, title=title, title_pad=title_pad)
    
    plt.xticks(np.linspace(0, points, ticks_x), np.linspace(0, x_lim * 100, ticks_x))
    plt.yticks(np.linspace(0, points, ticks_y), np.linspace(-p_lim, p_lim, ticks_y))
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=font_tick)
    ax.figure.axes[-1].yaxis.label.set_size(font_y_label)
    
    save_and_show_plt(plt, save_to_file)
   
def plot_both_models_with_input(models_DDQN, models_TD3, input_signal_data_DDQN, input_signal_data_TD3, save_to_file='', 
                                y_start_theta=None, y_end_theta=None, y_start_x=None, y_end_x=None, title_pad=35, 
                                size_x=48, size_y=36, dpi=200, linewidth=8, font_title=96, font_x_label=96, font_y_label=96, 
                                font_legend=96, font_tick=84, grid_linewidth=8, outline_width=8):
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.set_size_inches(size_x, size_y)
    fig.set_dpi(dpi)
    
    plot_model_with_input(models_DDQN, input_signal_data_DDQN, ax1, ax3, method='DDQN', title_pad=title_pad, 
                          y_start_theta=y_start_theta, y_end_theta=y_end_theta, y_start_x=y_start_x, y_end_x=y_end_x, 
                          linewidth=linewidth, font_title=font_title, font_x_label=font_x_label, font_y_label=font_y_label, 
                          font_tick=font_tick, grid_linewidth=grid_linewidth, outline_width=outline_width)
    
    plot_model_with_input(models_TD3, input_signal_data_TD3, ax2, ax4, method='TD3', color_start=len(models_DDQN), title_pad=title_pad, 
                          y_start_theta=y_start_theta, y_end_theta=y_end_theta, y_start_x=y_start_x, y_end_x=y_end_x, 
                          linewidth=linewidth, font_title=font_title, font_x_label=font_x_label, font_y_label=font_y_label, 
                          font_tick=font_tick, grid_linewidth=grid_linewidth, outline_width=outline_width)
    
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.36, wspace=0.20)
    
    save_and_show_plt(plt, save_to_file)
    
def plot_model_with_input(models, input_signal_data, ax1, ax2, method='', force_legend=False, color_start=0, 
                          y_start_theta=None, y_end_theta=None, y_start_x=None, y_end_x=None, title_pad=35, 
                          linewidth=8, font_title=96, font_x_label=96, font_y_label=96, font_legend=96, font_tick=88, 
                          grid_linewidth=8, outline_width=8):
    
    x_input, y_input, sim_time = input_signal_data
    
    max_t = 0.0
    
    for i in range(len(models)):
        if max(models[i].get_t()) > max_t:
            max_t = max(models[i].get_t())

        ax1.plot(models[i].get_t(), models[i].get_p_180(), linewidth=linewidth, color=f'C{i + 1 + color_start}')
        
        if i == (len(models) - 1):
            ax1.plot([0, sim_time], [0, 0], linewidth=linewidth, color='mediumblue')
            
        configure_axis(ax1, title=f'{method} Inclination Control', xlabel=r'$t$ (s)', ylabel=r'$\theta$ (°)', 
                       x_end=max_t, x_start=0, y_start=y_start_theta, y_end=y_end_theta, title_pad=title_pad, 
                       font_title=font_title, font_x_label=font_x_label, font_y_label=font_y_label, 
                       font_tick=font_tick, grid_linewidth=grid_linewidth, outline_width=outline_width)

        ax2.plot(models[i].get_t(), models[i].get_x() * 100, linewidth=linewidth, label=f'Agent {method} {i + 1}', color=f'C{i + 1 + color_start}')
        
        if i == (len(models) - 1):
            ax2.plot(x_input, np.array(y_input) * 100, linewidth=linewidth, color='mediumblue', label='Input Signal')
        
        configure_axis(ax2, title=f'{method} Position Control', xlabel=r'$t$ (s)', ylabel=r'$x$ (cm)', 
                       x_end=max_t, x_start=0, y_start=y_start_x, y_end=y_end_x, title_pad=title_pad, 
                       font_title=font_title, font_x_label=font_x_label, font_y_label=font_y_label, 
                       font_tick=font_tick, grid_linewidth=grid_linewidth, outline_width=outline_width)
        
        if force_legend:
            ax2.legend(fontsize=font_legend, loc='lower right')
        
        else:
            ax2.legend(fontsize=font_legend)
            
    ax1.tick_params(axis='both', which='major', pad=15)
    ax2.tick_params(axis='both', which='major', pad=15)

def plot_evaluation_histories(histories, save_to_file='', 
                              size_x=48, size_y=12, dpi=200, linewidth=4, markersize=24, 
                              font_title=56, font_x_label=56, font_y_label=56, font_tick=48, grid_linewidth=4, outline_width=4):
    
    run_lists = [i[0] for i in histories]
    score_mean_lists = [i[1] for i in histories]
    episode_lengths_mean_lists = [i[2] for i in histories]

    shorter_run_lists = run_lists[0]
    
    for i in range(1, len(histories)):
        if run_lists[i][-1] < shorter_run_lists[-1]:
            shorter_run_lists = run_lists[i]
            
    for i in range(len(histories)):
        run_lists[i] = run_lists[i][0:len(shorter_run_lists)]
        score_mean_lists[i] = score_mean_lists[i][0:len(shorter_run_lists)]
        episode_lengths_mean_lists[i] = episode_lengths_mean_lists[i][0:len(shorter_run_lists)]
            
    scores_mean = np.mean(score_mean_lists, axis=0)
    episode_length_mean = np.mean(episode_lengths_mean_lists, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2)
        
    fig.set_size_inches(size_x, size_y)
    fig.set_dpi(dpi)
    
    for i in range(len(histories)):
        ax1.plot(run_lists[i], -np.array(score_mean_lists[i]), 'o-', linewidth=linewidth, color='C0', markersize=markersize)
        ax2.plot(run_lists[i], episode_lengths_mean_lists[i], 'o-', linewidth=linewidth, color='C0', markersize=markersize)
        
    ax1.plot(shorter_run_lists, -scores_mean, 'o-', linewidth=linewidth, color='C1', markersize=markersize)
    ax2.plot(shorter_run_lists, episode_length_mean, 'o-', linewidth=linewidth, color='C1', markersize=markersize, label='Média de avaliações')
    
    configure_axis(ax1, yscale='log', title='Negative Score', xlabel='Episode', x_end=max(shorter_run_lists) + 10, 
                   font_title=font_title, font_x_label=font_x_label, font_y_label=font_y_label, 
                   font_tick=font_tick, grid_linewidth=grid_linewidth, outline_width=outline_width)

    configure_axis(ax2, yscale='log', title='Episode Length (s)', xlabel='Episode', x_end=max(shorter_run_lists) + 10, 
                    font_title=font_title, font_x_label=font_x_label, font_y_label=font_y_label, 
                    font_tick=font_tick, grid_linewidth=grid_linewidth, outline_width=outline_width)

    fig.tight_layout()
    
    save_and_show_plt(plt, save_to_file)
    
def plot_evaluation_histories_both(histories_DDQN, histories_TD3, save_to_file='', transparency=0.4, 
                                   size_x=24, size_y=24, dpi=200, linewidth=4, linewidth_mean=6, markersize=24, font_legend=48, 
                                   font_title=48, font_x_label=48, font_y_label=48, font_tick=44, grid_linewidth=4, outline_width=4):
    
    histories = histories_DDQN + histories_TD3
    run_lists = [i[0] for i in histories]
    score_mean_lists = [i[1] for i in histories]
    episode_lengths_mean_lists = [i[2] for i in histories]

    shorter_run_lists = run_lists[0]
    
    for i in range(1, len(histories)):
        if run_lists[i][-1] < shorter_run_lists[-1]:
            shorter_run_lists = run_lists[i]
            
    for i in range(len(histories)):
        run_lists[i] = run_lists[i][0:len(shorter_run_lists)]
        score_mean_lists[i] = score_mean_lists[i][0:len(shorter_run_lists)]
        episode_lengths_mean_lists[i] = episode_lengths_mean_lists[i][0:len(shorter_run_lists)]
            
    scores_mean_DDQN = np.mean(score_mean_lists[:len(histories_DDQN)], axis=0)
    scores_mean_TD3 = np.mean(score_mean_lists[len(histories_DDQN):], axis=0)
    episode_length_mean_DDQN = np.mean(episode_lengths_mean_lists[:len(histories_DDQN)], axis=0)
    episode_length_mean_TD3 = np.mean(episode_lengths_mean_lists[len(histories_DDQN):], axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1)
        
    fig.set_size_inches(size_x, size_y)
    fig.set_dpi(dpi)
    
    for i in range(len(histories)):
        if i < len(histories_DDQN):
            color = 'C0'
        else:
            color = 'C1'
            
        if i == 0:
            label = 'DDQN evaluations'
        elif i == len(histories_DDQN):
            label = 'TD3 evaluations'
        else:
            label = None
            
        ax1.plot(run_lists[i], -np.array(score_mean_lists[i]), '-', linewidth=linewidth, color=color, 
                 markersize=markersize, alpha=transparency, label=label)
        ax2.plot(run_lists[i], episode_lengths_mean_lists[i], '-', linewidth=linewidth, color=color, 
                 markersize=markersize, alpha=transparency, label=label)
        
    ax1.plot(shorter_run_lists, -scores_mean_DDQN, 'o-', linewidth=linewidth_mean, color='C0', markersize=markersize)
    ax1.plot(shorter_run_lists, -scores_mean_TD3, 'o-', linewidth=linewidth_mean, color='C1', markersize=markersize)
    ax2.plot(shorter_run_lists, episode_length_mean_DDQN, 'o-', linewidth=linewidth_mean, color='C0', markersize=markersize, 
             label='DDQN evaluations mean')
    ax2.plot(shorter_run_lists, episode_length_mean_TD3, 'o-', linewidth=linewidth_mean, color='C1', markersize=markersize, 
             label='TD3 evaluations mean')
    
    ax2.legend(loc='lower right', fontsize=font_legend)
        
    configure_axis(ax1, yscale='log', ylabel='Accumulated Cost', xlabel='Episode', x_end=max(shorter_run_lists) + 10, 
                   font_title=font_title, font_x_label=font_x_label, font_y_label=font_y_label, 
                   font_tick=font_tick, grid_linewidth=grid_linewidth, outline_width=outline_width)
    configure_axis(ax2, yscale='log', ylabel='Episode Length (s)', xlabel='Episode', x_end=max(shorter_run_lists) + 10, 
                    font_title=font_title, font_x_label=font_x_label, font_y_label=font_y_label, 
                    font_tick=font_tick, grid_linewidth=grid_linewidth, outline_width=outline_width)

    fig.tight_layout()
    plt.subplots_adjust(hspace=0.26)
    
    save_and_show_plt(plt, save_to_file)









