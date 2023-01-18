import matplotlib.pyplot as plt
import seaborn as sns
import transformers
model = transformers.AutoModel.from_pretrained('facebook/opt-125m')

# TODO, this plot shows the weights for an entire linear layer in the model. What we really need to do is plot the results for just a single channel, as that is what we are quantizing
# second, use the real max and min values that we use. (Here is fake max/min values)
data = model.decoder.layers[3].self_attn.k_proj.weight.data.view(-1).numpy()

from matplotlib import font_manager
font_path='/Users/ericwallace/Downloads/computer-modern/cmunss.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rc('axes', unicode_minus=False)
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

sns.histplot(data, color='#78b075', kde=True); plt.axvline(0.35, color='black', linestyle='--', ymax=0.5); plt.axvline(-0.35, color='black', linestyle='--', ymax=0.5); plt.text(0.355, 750, 'Max Weight Threshold', rotation=270, color='black', style='italic'); plt.text(-0.38, 750, 'Min Weight Threshold', rotation=90, color='black', style='italic'); plt.annotate('Max',xy=(max(data), 0), xytext=(max(data), 1000), horizontalalignment="center", arrowprops=dict(arrowstyle='->',lw=1)); plt.annotate('Min',xy=(min(data), 0), xytext=(min(data), 1000), horizontalalignment="center", arrowprops=dict(arrowstyle='->',lw=1)); plt.ylim((0,11000));
plt.savefig('clipping.pdf')
# plt.title('Weight Distribution for an OPT Layer');
