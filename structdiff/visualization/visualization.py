# structdiff/visualization/visualization.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from Bio import PDB
import py3Dmol
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import networkx as nx
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PeptideVisualizer:
    """Comprehensive visualization tools for peptide analysis"""
    
    def __init__(self, style: str = 'publication'):
        self.style = style
        self._setup_style()
    
    def _setup_style(self):
        """Setup visualization style"""
        if self.style == 'publication':
            plt.style.use('seaborn-v0_8-paper')
            sns.set_palette("husl")
            plt.rcParams.update({
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'figure.figsize': (8, 6),
                'figure.dpi': 150
            })
        elif self.style == 'presentation':
            plt.style.use('dark_background')
            plt.rcParams.update({
                'font.size': 16,
                'axes.labelsize': 18,
                'axes.titlesize': 20,
                'figure.figsize': (12, 8),
                'figure.dpi': 100
            })
    
    def plot_sequence_logo(
        self,
        sequences: List[str],
        title: str = "Sequence Logo",
        save_path: Optional[str] = None
    ):
        """Create sequence logo from aligned sequences"""
        from logomaker import Logo, alignment_to_matrix
        
        # Convert to DataFrame for logomaker
        max_len = max(len(seq) for seq in sequences)
        
        # Pad sequences
        padded_seqs = []
        for seq in sequences:
            padded = seq + '-' * (max_len - len(seq))
            padded_seqs.append(list(padded))
        
        # Create position frequency matrix
        seq_df = pd.DataFrame(padded_seqs)
        counts_mat = alignment_to_matrix(seq_df)
        
        # Create logo
        fig, ax = plt.subplots(figsize=(max_len * 0.5, 4))
        logo = Logo(counts_mat, ax=ax)
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Information Content')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_property_distribution(
        self,
        sequences: List[str],
        properties: List[str] = ['charge', 'hydrophobicity', 'length'],
        save_path: Optional[str] = None
    ):
        """Plot distribution of sequence properties"""
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        
        # Calculate properties
        data = {prop: [] for prop in properties}
        
        for seq in sequences:
            if 'length' in properties:
                data['length'].append(len(seq))
            
            if any(p in properties for p in ['charge', 'hydrophobicity', 'instability']):
                try:
                    analyzed = ProteinAnalysis(seq)
                    
                    if 'charge' in properties:
                        data['charge'].append(analyzed.charge_at_pH(7.0))
                    
                    if 'hydrophobicity' in properties:
                        data['hydrophobicity'].append(analyzed.gravy())
                    
                    if 'instability' in properties:
                        data['instability'].append(analyzed.instability_index())
                except:
                    pass
        
        # Create subplots
        n_props = len(properties)
        fig, axes = plt.subplots(1, n_props, figsize=(5*n_props, 5))
        
        if n_props == 1:
            axes = [axes]
        
        for i, (prop, values) in enumerate(data.items()):
            if values:
                axes[i].hist(values, bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_xlabel(prop.capitalize())
                axes[i].set_ylabel('Count')
                axes[i].set_title(f'{prop.capitalize()} Distribution')
                
                # Add statistics
                mean_val = np.mean(values)
                std_val = np.std(values)
                axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2)
                axes[i].text(
                    0.7, 0.9,
                    f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                    transform=axes[i].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_embedding_space(
        self,
        embeddings: torch.Tensor,
        labels: Optional[List[str]] = None,
        method: str = 'tsne',
        perplexity: int = 30,
        n_components: int = 2,
        save_path: Optional[str] = None
    ):
        """Visualize high-dimensional embeddings"""
        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        # Reduce dimensionality
        if method == 'tsne':
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=n_components)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        reduced = reducer.fit_transform(embeddings)
        
        # Create plot
        if n_components == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            if labels is not None:
                # Color by labels
                unique_labels = list(set(labels))
                colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    mask = [l == label for l in labels]
                    ax.scatter(
                        reduced[mask, 0],
                        reduced[mask, 1],
                        c=[colors[i]],
                        label=label,
                        alpha=0.7,
                        s=50
                    )
                
                ax.legend()
            else:
                ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, s=50)
            
            ax.set_xlabel(f'{method.upper()} 1')
            ax.set_ylabel(f'{method.upper()} 2')
            ax.set_title(f'Embedding Space Visualization ({method.upper()})')
            
        elif n_components == 3:
            # 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            if labels is not None:
                unique_labels = list(set(labels))
                colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    mask = [l == label for l in labels]
                    ax.scatter(
                        reduced[mask, 0],
                        reduced[mask, 1],
                        reduced[mask, 2],
                        c=[colors[i]],
                        label=label,
                        alpha=0.7,
                        s=50
                    )
                
                ax.legend()
            else:
                ax.scatter(
                    reduced[:, 0],
                    reduced[:, 1],
                    reduced[:, 2],
                    alpha=0.7,
                    s=50
                )
            
            ax.set_xlabel(f'{method.upper()} 1')
            ax.set_ylabel(f'{method.upper()} 2')
            ax.set_zlabel(f'{method.upper()} 3')
            ax.set_title(f'3D Embedding Space ({method.upper()})')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        sequence: str,
        layer: int = -1,
        head: Optional[int] = None,
        save_path: Optional[str] = None
    ):
        """Plot attention weights as heatmap"""
        # Get specific layer
        if attention_weights.dim() == 4:  # (layers, heads, seq, seq)
            attn = attention_weights[layer]
        else:
            attn = attention_weights
        
        # Average over heads or select specific head
        if head is not None:
            attn = attn[head]
        else:
            attn = attn.mean(dim=0)
        
        # Convert to numpy
        attn = attn.cpu().numpy()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(attn, cmap='Blues', aspect='auto')
        
        # Set ticks
        positions = list(range(len(sequence)))
        ax.set_xticks(positions)
        ax.set_yticks(positions)
        ax.set_xticklabels(list(sequence), rotation=45)
        ax.set_yticklabels(list(sequence))
        
        # Labels
        ax.set_xlabel('Target Position')
        ax.set_ylabel('Source Position')
        ax.set_title(f'Attention Weights (Layer {layer}' + 
                    (f', Head {head}' if head is not None else ', Averaged') + ')')
        
        # Colorbar
        plt.colorbar(im, ax=ax)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_structure_3d(
        self,
        pdb_path: str,
        color_scheme: str = 'secondary',
        save_path: Optional[str] = None
    ):
        """Interactive 3D structure visualization"""
        # Load structure
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('peptide', pdb_path)
        
        # Create py3Dmol viewer
        view = py3Dmol.view(width=800, height=600)
        
        # Add structure
        with open(pdb_path, 'r') as f:
            pdb_string = f.read()
        
        view.addModel(pdb_string, 'pdb')
        
        # Set style based on color scheme
        if color_scheme == 'secondary':
            view.setStyle({'cartoon': {'color': 'spectrum'}})
        elif color_scheme == 'residue':
            view.setStyle({'stick': {'colorscheme': 'amino'}})
        elif color_scheme == 'hydrophobicity':
            view.setStyle({'surface': {'colorscheme': {
                'prop': 'b',
                'gradient': 'rwb'
            }}})
        
        view.zoomTo()
        
        if save_path:
            # Save as HTML
            html = view._make_html()
            with open(save_path, 'w') as f:
                f.write(html)
        
        return view
    
    def plot_diffusion_trajectory(
        self,
        trajectory: torch.Tensor,
        timesteps: Optional[List[int]] = None,
        reduce_dim: bool = True,
        save_path: Optional[str] = None
    ):
        """Visualize diffusion denoising trajectory"""
        # trajectory shape: (T, B, L, D)
        T, B, L, D = trajectory.shape
        
        if timesteps is None:
            timesteps = list(range(T))
        
        # Flatten to (T*B, L*D) for dimension reduction
        if reduce_dim:
            trajectory_flat = trajectory.reshape(T * B, -1).cpu().numpy()
            
            # PCA for visualization
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(trajectory_flat)
            reduced = reduced.reshape(T, B, 2)
        
        # Create interactive plot with plotly
        fig = go.Figure()
        
        # Add traces for each sequence in batch
        colors = px.colors.qualitative.Plotly
        
        for b in range(min(B, 10)):  # Limit to 10 sequences
            if reduce_dim:
                x_vals = reduced[:, b, 0]
                y_vals = reduced[:, b, 1]
            else:
                # Use first two dimensions
                x_vals = trajectory[:, b, 0, 0].cpu().numpy()
                y_vals = trajectory[:, b, 0, 1].cpu().numpy()
            
            # Add trajectory
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                name=f'Sequence {b+1}',
                line=dict(color=colors[b % len(colors)], width=2),
                marker=dict(size=8)
            ))
            
            # Add timestep labels
            for i in range(0, T, max(1, T // 10)):
                fig.add_annotation(
                    x=x_vals[i],
                    y=y_vals[i],
                    text=f't={timesteps[i]}',
                    showarrow=False,
                    font=dict(size=8)
                )
        
        fig.update_layout(
            title='Diffusion Denoising Trajectory',
            xaxis_title='PC1' if reduce_dim else 'Dim 1',
            yaxis_title='PC2' if reduce_dim else 'Dim 2',
            hovermode='closest',
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def plot_generation_metrics(
        self,
        metrics_history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ):
        """Plot training/generation metrics over time"""
        n_metrics = len(metrics_history)
        
        fig = make_subplots(
            rows=(n_metrics + 1) // 2,
            cols=2,
            subplot_titles=list(metrics_history.keys())
        )
        
        for i, (metric_name, values) in enumerate(metrics_history.items()):
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    mode='lines',
                    name=metric_name
                ),
                row=row,
                col=col
            )
            
            fig.update_xaxes(title_text="Step", row=row, col=col)
            fig.update_yaxes(title_text=metric_name, row=row, col=col)
        
        fig.update_layout(
            height=300 * ((n_metrics + 1) // 2),
            showlegend=False,
            title_text="Training Metrics"
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def plot_sequence_similarity_network(
        self,
        sequences: List[str],
        labels: Optional[List[str]] = None,
        similarity_threshold: float = 0.7,
        save_path: Optional[str] = None
    ):
        """Create network visualization of sequence similarities"""
        from Bio import pairwise2
        
        # Compute pairwise similarities
        n_seqs = len(sequences)
        similarity_matrix = np.zeros((n_seqs, n_seqs))
        
        for i in range(n_seqs):
            for j in range(i+1, n_seqs):
                alignment = pairwise2.align.globalxx(sequences[i], sequences[j])[0]
                similarity = alignment.score / max(len(sequences[i]), len(sequences[j]))
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Create network
        G = nx.Graph()
        
        # Add nodes
        for i, seq in enumerate(sequences):
            G.add_node(i, sequence=seq, label=labels[i] if labels else str(i))
        
        # Add edges for similar sequences
        for i in range(n_seqs):
            for j in range(i+1, n_seqs):
                if similarity_matrix[i, j] >= similarity_threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i, j])
        
        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Draw nodes
        if labels:
            unique_labels = list(set(labels))
            colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_labels)))
            node_colors = [colors[unique_labels.index(labels[i])] for i in range(n_seqs)]
        else:
            node_colors = 'lightblue'
        
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=500,
            alpha=0.7
        )
        
        # Draw edges
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(
            G, pos,
            width=[w * 3 for w in weights],
            alpha=[w * 0.7 for w in weights]
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            {i: G.nodes[i]['label'] for i in G.nodes()},
            font_size=8
        )
        
        plt.title(f'Sequence Similarity Network (threshold={similarity_threshold})')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_report(
        self,
        sequences: List[str],
        metrics: Dict[str, float],
        output_path: str = "peptide_generation_report.html"
    ):
        """Create comprehensive HTML report"""
        from jinja2 import Template
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Peptide Generation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                h2 { color: #666; }
                .metric { background: #f0f0f0; padding: 10px; margin: 10px 0; }
                .sequence { font-family: monospace; background: #e0e0e0; padding: 5px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
            </style>
        </head>
        <body>
            <h1>Peptide Generation Report</h1>
            
            <h2>Summary Statistics</h2>
            <div class="metric">
                <strong>Total Sequences:</strong> {{ n_sequences }}<br>
                <strong>Average Length:</strong> {{ avg_length }}<br>
                <strong>Unique Sequences:</strong> {{ n_unique }}
            </div>
            
            <h2>Generation Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                {% for metric, value in metrics.items() %}
                <tr>
                    <td>{{ metric }}</td>
                    <td>{{ "%.4f"|format(value) }}</td>
                </tr>
                {% endfor %}
            </table>
            
            <h2>Sample Sequences</h2>
            {% for seq in sample_sequences[:10] %}
            <div class="sequence">{{ seq }}</div>
            {% endfor %}
            
            <h2>Length Distribution</h2>
            <img src="length_distribution.png" width="600">
            
            <h2>Property Analysis</h2>
            <img src="property_distribution.png" width="800">
            
            <h2>Sequence Logo</h2>
            <img src="sequence_logo.png" width="800">
            
            <p><em>Generated on {{ timestamp }}</em></p>
        </body>
        </html>
        """
        
        # Generate plots
        self.plot_property_distribution(sequences, save_path="property_distribution.png")
        self.plot_sequence_logo(sequences[:50], save_path="sequence_logo.png")
        
        # Calculate statistics
        lengths = [len(seq) for seq in sequences]
        
        # Render template
        template = Template(html_template)
        html_content = template.render(
            n_sequences=len(sequences),
            avg_length=np.mean(lengths),
            n_unique=len(set(sequences)),
            metrics=metrics,
            sample_sequences=sequences,
            timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Report saved to {output_path}")


# Interactive visualization app
class InteractivePeptideExplorer:
    """Interactive dashboard for exploring generated peptides"""
    
    def __init__(self, sequences: List[str], embeddings: Optional[torch.Tensor] = None):
        self.sequences = sequences
        self.embeddings = embeddings
        self._compute_properties()
    
    def _compute_properties(self):
        """Compute peptide properties for visualization"""
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        
        self.properties_df = []
        
        for seq in self.sequences:
            try:
                analyzed = ProteinAnalysis(seq)
                props = {
                    'sequence': seq,
                    'length': len(seq),
                    'charge': analyzed.charge_at_pH(7.0),
                    'hydrophobicity': analyzed.gravy(),
                    'molecular_weight': analyzed.molecular_weight(),
                    'instability': analyzed.instability_index(),
                    'aromaticity': analyzed.aromaticity()
                }
                self.properties_df.append(props)
            except:
                pass
        
        self.properties_df = pd.DataFrame(self.properties_df)
    
    def create_dashboard(self):
        """Create interactive Plotly dashboard"""
        from dash import Dash, html, dcc, Input, Output
        import plotly.express as px
        
        app = Dash(__name__)
        
        # Layout
        app.layout = html.Div([
            html.H1("Peptide Explorer Dashboard"),
            
            html.Div([
                html.Div([
                    html.Label("X-axis property:"),
                    dcc.Dropdown(
                        id='x-property',
                        options=[{'label': col, 'value': col} 
                                for col in self.properties_df.columns if col != 'sequence'],
                        value='hydrophobicity'
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Y-axis property:"),
                    dcc.Dropdown(
                        id='y-property',
                        options=[{'label': col, 'value': col} 
                                for col in self.properties_df.columns if col != 'sequence'],
                        value='charge'
                    )
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ]),
            
            dcc.Graph(id='scatter-plot'),
            
            html.Div(id='sequence-display')
        ])
        
        # Callbacks
        @app.callback(
            Output('scatter-plot', 'figure'),
            [Input('x-property', 'value'),
             Input('y-property', 'value')]
        )
        def update_scatter(x_prop, y_prop):
            fig = px.scatter(
                self.properties_df,
                x=x_prop,
                y=y_prop,
                hover_data=['sequence'],
                title=f'{y_prop} vs {x_prop}'
            )
            return fig
        
        return app
# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04
