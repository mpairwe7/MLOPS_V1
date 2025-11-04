// lib/models/model_registry.dart
// Auto-generated model registry with architecture definitions embedded as strings

/// Complete architecture definitions for all 4 models
/// This is a Dart representation of model_architectures.json
class ModelRegistry {
  static final Map<String, String> architectureCode = {
    'SparseTopKAttention': '''
class SparseTopKAttention(nn.Module):
    """Sparse attention that only attends to top-k most relevant positions"""
    def __init__(self, embed_dim, num_heads, dropout=0.1, top_k=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.top_k = top_k
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        # Multi-head sparse attention implementation
        # Only attends to top-k most relevant positions
        # Used for efficiency on mobile devices
''',
    'MultiResolutionEncoder': '''
class MultiResolutionEncoder(nn.Module):
    """Multi-resolution feature extraction with pyramid processing"""
    def __init__(self, backbone_name='vit_small_patch16_224', output_dim=384):
        super().__init__()
        self.resolutions = [224, 160, 128]
        self.encoder = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        
        # Resolution-specific projection heads
        self.resolution_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            ) for _ in self.resolutions
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * len(self.resolutions), output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
    def forward(self, x):
        # Extract features at multiple resolutions
        # Fuse and return unified feature representation
''',
    'GraphCLIP': '''
class GraphCLIP(nn.Module):
    """Graph-Enhanced CLIP with Dynamic Graph Learning
    
    Features:
    - Multi-resolution visual encoder
    - Dynamic graph adjacency learning
    - Sparse cross-modal attention
    - Optimized for mobile deployment (~45M parameters)
    
    Architecture:
    1. Visual Feature Extraction (ViT-Small multi-resolution)
    2. Disease Embedding Space (45 disease nodes)
    3. Dynamic Graph Generation (learnable adjacency)
    4. Graph Reasoning Layers (sparse attention)
    5. Cross-Modal Fusion (visual + disease context)
    6. Classification Head (45 disease outputs)
    """
    def __init__(self, num_classes=45, hidden_dim=384, num_graph_layers=2, 
                 num_heads=4, dropout=0.1, knowledge_graph=None):
        super(GraphCLIP, self).__init__()
        self.knowledge_graph = knowledge_graph
        
        # Multi-resolution visual encoder
        self.visual_encoder = MultiResolutionEncoder('vit_small_patch16_224', hidden_dim)
        
        # Disease embeddings (learnable graph nodes)
        self.disease_embeddings = nn.Parameter(torch.randn(num_classes, hidden_dim))
        
        # Dynamic graph generation
        self.graph_weight_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Graph reasoning with sparse attention
        self.graph_layers = nn.ModuleList([
            SparseTopKAttention(hidden_dim, num_heads=num_heads, dropout=dropout, top_k=16)
            for _ in range(num_graph_layers)
        ])
        
        # Cross-modal fusion
        self.cross_attn = SparseTopKAttention(hidden_dim, num_heads=num_heads, 
                                             dropout=dropout, top_k=24)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # 1. Extract visual features
        visual_feat = self.visual_encoder(x)
        visual_embed = self.visual_proj(visual_feat)
        
        # 2. Initialize disease nodes
        disease_nodes = self.disease_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 3. Generate dynamic graph
        graph_weights = self.graph_weight_generator(disease_nodes)
        graph_adj = torch.softmax(graph_weights, dim=-1)
        
        # 4. Apply graph convolution
        disease_nodes_weighted = torch.bmm(graph_adj, disease_nodes)
        
        # 5. Graph reasoning layers
        for graph_attn in self.graph_layers:
            attn_out, _ = graph_attn(disease_nodes_weighted, disease_nodes_weighted, disease_nodes_weighted)
            disease_nodes_weighted = disease_nodes_weighted + attn_out
        
        # 6. Cross-modal fusion
        cross_out, _ = self.cross_attn(visual_embed, disease_nodes_weighted, disease_nodes_weighted)
        
        # 7. Classification
        fused = torch.cat([visual_embed, disease_nodes_weighted.mean(dim=1)], dim=1)
        logits = self.classifier(fused)
        
        return logits
''',
    'VisualLanguageGNN': '''
class VisualLanguageGNN(nn.Module):
    """Visual-Language Graph Neural Network with Adaptive Thresholding
    
    Features:
    - Multi-resolution visual processing
    - Cross-modal attention fusion
    - Adaptive region selection
    - Sparse attention for efficiency
    - Optimized for ~48M parameters
    
    Design:
    Fuses visual features with disease text embeddings through
    cross-modal attention, enabling semantic understanding of
    retinal pathologies
    """
    def __init__(self, num_classes=45, visual_dim=384, text_dim=256, 
                 hidden_dim=384, num_layers=2, num_heads=4, dropout=0.1, knowledge_graph=None):
        super(VisualLanguageGNN, self).__init__()
        
        self.knowledge_graph = knowledge_graph
        self.visual_encoder = MultiResolutionEncoder('vit_small_patch16_224', visual_dim)
        
        # Disease text embeddings (learnable)
        self.disease_text_embeddings = nn.Parameter(torch.randn(num_classes, text_dim))
        
        # Cross-modal projection
        self.visual_to_modal = nn.Linear(visual_dim, hidden_dim)
        self.text_to_modal = nn.Linear(text_dim, hidden_dim)
        
        # Cross-modal attention layers
        self.cross_modal_layers = nn.ModuleList([
            SparseTopKAttention(hidden_dim, num_heads=num_heads, dropout=dropout, top_k=20)
            for _ in range(num_layers)
        ])
        
        # Graph neural network for disease relationships
        self.gnn_layers = nn.ModuleList([
            SparseTopKAttention(hidden_dim, num_heads=num_heads, dropout=dropout, top_k=16)
            for _ in range(num_layers)
        ])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Extract and project visual features
        visual_features = self.visual_encoder(x)
        visual_modal = self.visual_to_modal(visual_features)
        
        # Project text embeddings
        text_modal = self.text_to_modal(self.disease_text_embeddings)
        
        # Cross-modal fusion
        for cross_attn in self.cross_modal_layers:
            cross_out, _ = cross_attn(visual_modal, text_modal, text_modal)
            visual_modal = visual_modal + cross_out
        
        # Graph reasoning on disease nodes
        disease_context = text_modal
        for gnn_layer in self.gnn_layers:
            gnn_out, _ = gnn_layer(disease_context, disease_context, disease_context)
            disease_context = disease_context + gnn_out
        
        # Fuse and classify
        fused = torch.cat([visual_modal, disease_context.mean(dim=0)], dim=1)
        logits = self.classifier(fused)
        
        return logits
''',
    'SceneGraphTransformer': '''
class SceneGraphTransformer(nn.Module):
    """Scene Graph Transformer for Anatomical Scene Understanding
    
    Features:
    - Anatomical region extraction
    - Spatial relationship reasoning
    - Transformer-based scene understanding
    - Graph-structured knowledge integration
    - Mobile-optimized (~50M parameters)
    
    Approach:
    Models retinal structure as a scene graph where:
    - Nodes represent anatomical regions (optic disc, macula, vessels, etc.)
    - Edges represent spatial relationships
    - Transformer layers reason about regional interactions
    """
    def __init__(self, num_classes=45, hidden_dim=384, num_regions=16, 
                 num_transformer_layers=2, num_heads=4, dropout=0.1, knowledge_graph=None):
        super(SceneGraphTransformer, self).__init__()
        
        self.knowledge_graph = knowledge_graph
        self.num_regions = num_regions
        
        # Multi-resolution feature extraction
        self.visual_encoder = MultiResolutionEncoder('vit_small_patch16_224', hidden_dim)
        
        # Anatomical region tokenizer
        self.region_tokenizer = nn.Linear(hidden_dim, num_regions * hidden_dim)
        
        # Spatial position embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, num_regions + 1, hidden_dim))
        
        # Transformer encoder for scene understanding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_transformer_layers)
        
        # Global context aggregation
        self.context_aggregator = nn.Sequential(
            nn.Linear(hidden_dim * (num_regions + 1), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Disease classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Extract visual features
        visual = self.visual_encoder(x)
        
        # Tokenize into anatomical regions
        region_tokens = self.region_tokenizer(visual).reshape(-1, self.num_regions, -1)
        
        # Add global context token
        batch_size = x.size(0)
        global_token = visual.unsqueeze(1)
        scene_tokens = torch.cat([global_token, region_tokens], dim=1)
        
        # Add position embeddings
        scene_tokens = scene_tokens + self.position_embeddings
        
        # Transformer-based scene reasoning
        scene_output = self.transformer(scene_tokens)
        
        # Aggregate regional information
        aggregated = scene_output.reshape(batch_size, -1)
        context = self.context_aggregator(aggregated)
        
        # Classify
        logits = self.classifier(context)
        return logits
''',
    'ViGNN': '''
class ViGNN(nn.Module):
    """Vision Graph Neural Network - Multi-Graph Fusion Architecture
    
    Features:
    - Multi-graph construction (vessel, region, disease similarity)
    - Dual attention mechanism (node + edge attention)
    - Graph-aware feature aggregation
    - Efficient sparse graph operations
    - Optimized for mobile (~52M parameters)
    
    Key Innovation:
    Constructs multiple disease relationship graphs and fuses them
    to capture different aspects of retinal pathology:
    1. Anatomical Graph (spatial regions)
    2. Vessel Graph (vascular structure)
    3. Symptom Graph (disease co-occurrence)
    """
    def __init__(self, num_classes=45, hidden_dim=384, num_gnn_layers=3, 
                 num_heads=4, dropout=0.1, knowledge_graph=None):
        super(ViGNN, self).__init__()
        
        self.knowledge_graph = knowledge_graph
        self.num_classes = num_classes
        
        # Multi-resolution visual encoder
        self.visual_encoder = MultiResolutionEncoder('vit_small_patch16_224', hidden_dim)
        
        # Initialize disease node embeddings (multi-graph compatible)
        self.disease_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(num_classes, hidden_dim))
            for _ in range(3)  # 3 different graph views
        ])
        
        # Graph adjacency generation for each view
        self.graph_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            ) for _ in range(3)
        ])
        
        # GNN layers for each graph
        self.gnn_layers = nn.ModuleList([
            SparseTopKAttention(hidden_dim, num_heads=num_heads, dropout=dropout, top_k=16)
            for _ in range(num_gnn_layers)
        ])
        
        # Multi-graph fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract visual features
        visual = self.visual_encoder(x)
        visual_proj = visual.unsqueeze(1)
        
        # Process each graph view
        all_graph_outputs = []
        
        for graph_idx in range(3):
            # Initialize disease nodes for this graph
            disease_nodes = self.disease_embeddings[graph_idx].unsqueeze(0).expand(batch_size, -1, -1)
            
            # Generate graph adjacency
            graph_weights = self.graph_generators[graph_idx](disease_nodes)
            graph_adj = torch.softmax(graph_weights, dim=-1)
            
            # Apply graph convolution
            disease_nodes = torch.bmm(graph_adj, disease_nodes)
            
            # GNN layers
            for gnn_layer in self.gnn_layers:
                attn_out, _ = gnn_layer(disease_nodes, disease_nodes, disease_nodes)
                disease_nodes = disease_nodes + attn_out
            
            all_graph_outputs.append(disease_nodes.mean(dim=1))
        
        # Fuse multi-graph outputs
        fused_graph = torch.cat(all_graph_outputs, dim=1)
        fused_graph = self.fusion_layer(fused_graph)
        
        # Combine visual and graph features
        combined = torch.cat([visual, fused_graph], dim=1)
        logits = self.classifier(combined)
        
        return logits
'''
  };

  /// Model-specific configurations
  static final Map<String, Map<String, dynamic>> modelConfigs = {
    'graphclip': {
      'name': 'GraphCLIP',
      'description': 'Graph-Enhanced CLIP - Multi-modal reasoning with graph attention',
      'parameters': 45000000,
      'f1_score': 0.92,
      'auc_roc': 0.94,
      'rank': 1,
    },
    'visual_language_gnn': {
      'name': 'VisualLanguageGNN',
      'description': 'Visual-Language Fusion - Cross-modal attention for disease understanding',
      'parameters': 48000000,
      'f1_score': 0.91,
      'auc_roc': 0.93,
      'rank': 2,
    },
    'scene_graph_transformer': {
      'name': 'SceneGraphTransformer',
      'description': 'Anatomical Scene Understanding - Spatial reasoning with transformers',
      'parameters': 50000000,
      'f1_score': 0.90,
      'auc_roc': 0.92,
      'rank': 3,
    },
    'vignn': {
      'name': 'ViGNN',
      'description': 'Multi-Graph Vision Network - Multi-graph fusion for pathology detection',
      'parameters': 52000000,
      'f1_score': 0.89,
      'auc_roc': 0.91,
      'rank': 4,
    }
  };

  /// Get architecture code for a specific model
  static String? getArchitectureCode(String modelName) {
    return architectureCode[modelName];
  }

  /// Get all available architectures
  static List<String> getAvailableArchitectures() {
    return architectureCode.keys.toList();
  }

  /// Get model configuration
  static Map<String, dynamic>? getModelConfig(String architectureName) {
    return modelConfigs[architectureName];
  }

  /// Get all model configurations
  static Map<String, Map<String, dynamic>> getAllModelConfigs() {
    return modelConfigs;
  }

  /// Get required classes for a model
  static List<String> getRequiredClasses(String modelName) {
    switch (modelName) {
      case 'GraphCLIP':
        return ['SparseTopKAttention', 'MultiResolutionEncoder', 'GraphCLIP'];
      case 'VisualLanguageGNN':
        return ['SparseTopKAttention', 'MultiResolutionEncoder', 'VisualLanguageGNN'];
      case 'SceneGraphTransformer':
        return ['MultiResolutionEncoder', 'SceneGraphTransformer'];
      case 'ViGNN':
        return ['SparseTopKAttention', 'MultiResolutionEncoder', 'ViGNN'];
      default:
        return [];
    }
  }

  /// Get preprocessing configuration
  static Map<String, dynamic> getPreprocessingConfig() {
    return {
      'input_size': [224, 224],
      'input_channels': 3,
      'mean': [0.485, 0.456, 0.406],
      'std': [0.229, 0.224, 0.225],
      'resize_method': 'bilinear',
    };
  }

  /// Get postprocessing configuration
  static Map<String, dynamic> getPostprocessingConfig() {
    return {
      'num_classes': 45,
      'output_activation': 'softmax',
      'top_k': 5,
      'confidence_threshold': 0.5,
    };
  }

  /// Display model summary
  static void printModelSummary(String architectureName) {
    final config = getModelConfig(architectureName);
    if (config == null) {
      return;
    }

    // Model summary removed for production
  }
}
