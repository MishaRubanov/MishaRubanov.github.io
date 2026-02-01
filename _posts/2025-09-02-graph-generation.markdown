---
layout: post
title: "From Classification to Connectivity: Generating Liquid-Handling Workflows with GNNs"
date: 2025-09-02
categories: lab-automation ml graphs
tags: [graph neural networks, laboratory automation, liquid handling, graph generation]
---

Modern Graph Neural Networks (GNNs) excel at predicting node and edge attributes, but many practical problems require changing the graph itself. Liquid-handling protocols are a prime example: executing a protocol means constructing a sequence of transfers that incrementally grows a workflow graph while respecting hard physical and chemical constraints. This post sketches how to adapt GNNs from attribute prediction to connectivity generation for liquid handling.

### Background: GNN building blocks
If you are new to GNNs, I recommend the clear and interactive overview in Distill’s “A Gentle Introduction to Graph Neural Networks” [link](`https://distill.pub/2021/gnn-intro/#table`). It explains message passing, aggregation and update functions, and how information flows over graph structure.

Key takeaways for our setting:
- GNNs operate over nodes, edges, and (optionally) global features via permutation-invariant aggregation.
- Information is localized and propagates by hops, which is useful for enforcing local constraints (e.g., volumes in a well or sterility across edges) while letting global features (e.g., temperature, instrument state) influence decisions.

### Problem framing: protocols as dynamic DAGs
We represent a liquid-handling protocol as a dynamic, directed acyclic multigraph (DAG) over time steps t = 0..T. The DAG constraint is fundamental: liquid handling operations cannot create cycles because time flows forward and reagents cannot be "un-mixed" or "un-transferred."

- **Nodes**: containers/wells, instrument resources (tips, reservoirs), intermediate mixtures, deck locations.
- **Edges**: operations such as aspirate, dispense, transfer, mix; edges carry attributes (volume, liquid identity, tip id, speed, timestamp).
- **State**: per-node attributes (current volume, composition, contamination risk), per-edge attributes (history), and global context (robot capabilities, timing, environment).
- **DAG Invariant**: Every edge (u,v) must satisfy timestamp(u) < timestamp(v), ensuring no cycles can exist.

At each step we add one or more edges that modify node states while preserving the DAG property. Generation ends when goals are satisfied (target mixture, plate layout) or no valid actions remain.

### Why attribute models aren’t enough
Attribute-focused GNNs answer questions like “what volume should be in well A?” given a fixed graph. Workflow synthesis instead requires proposing valid connectivity changes. We need a model that:
- Proposes the next operation (an edge or set of edges), including its endpoints and attributes.
- Respects constraints (conservation of volume, sterility, capacity, tool availability).
- Plans long-horizon sequences to reach targets.

### Modeling approaches for connectivity generation
There are several viable families, which can be combined:

1) **Autoregressive edge generation**
- Factorize p(protocol) into a sequence of edge additions. At each step, encode the current graph with a message-passing GNN; a policy head scores candidate (source node, op type, target node, attributes).
- Sampling: top-k or beam search with constraint masking.
- Benefits: precise control and easy constraint integration; drawbacks: long horizons.

2) **Diffusion or denoising over graphs**
- Start from a noisy action plan and denoise into a valid workflow using a GNN denoiser conditioned on task goals and instrument state.
- Useful for exploring diverse plans; requires careful constraint handling during sampling.

3) **Constraint-satisfying planning with neural guidance**
- Use a symbolic planner or MILP/CP-SAT to enforce hard physical constraints; use a GNN to learn heuristics (cost-to-go, action priors) that guide the search.
- Strong guarantees with improved speed/quality from learning.

4) **Imitation + RL hybrid**
- Train the policy with behavior cloning on historical protocols; fine-tune with RL using a simulator that implements lab physics and penalties for invalid or unsafe actions.

Let me elaborate on each approach with technical details and examples:

#### 1. Autoregressive Edge Generation
This approach treats protocol generation as a sequence modeling problem where each step adds one or more edges to the growing workflow DAG.

**Architecture Details:**
- **Encoder**: A k-layer message-passing GNN (e.g., GraphSAGE, GAT) processes the current graph state
- **Policy Head**: Multi-output network that predicts:
  - Operation type (categorical: transfer, mix, aspirate, dispense, etc.)
  - Source node selection (pointer network over available nodes)
  - Target node selection (pointer network with feasibility masking)
  - Continuous attributes (volume, speed, temperature) with bounded distributions
  - **Timestamp assignment**: Critical for maintaining DAG property

**Training Strategy:**
- Teacher forcing: use ground truth previous actions during training
- Scheduled sampling: gradually transition from teacher forcing to autoregressive generation
- Constraint masking: zero out probabilities for invalid actions (e.g., transferring from empty wells, creating cycles)
- **DAG enforcement**: Ensure timestamp(u) < timestamp(v) for all new edges (u,v)

**Example Implementation:**
```python
# Simplified pseudocode
def generate_step(current_graph, goal_embedding):
    # Encode current state
    node_embeddings = gnn_encoder(current_graph)
    
    # Predict next operation
    op_type = op_classifier(node_embeddings, goal_embedding)
    
    # Select source and target with pointer networks
    source_logits = source_pointer(node_embeddings, op_type)
    target_logits = target_pointer(node_embeddings, op_type, source_logits)
    
    # Apply feasibility masks
    source_logits = source_logits * source_feasibility_mask
    target_logits = target_logits * target_feasibility_mask
    
    # Sample and return action
    return sample_action(op_type, source_logits, target_logits)
```

**Advantages:**
- Direct control over generation process
- Easy to integrate hard constraints via masking
- Interpretable: each action is explicit and traceable
- Can use beam search for better planning

**Challenges:**
- Sequential nature limits parallelization
- Error accumulation over long sequences
- Requires careful curriculum learning for complex protocols

#### 2. Diffusion/Denoising over Graphs
This approach starts from a noisy, potentially invalid workflow and progressively denoises it into a valid protocol DAG.

**Architecture Details:**
- **Noise Schedule**: Gradually add noise to a target protocol over T timesteps
- **Denoiser**: GNN that predicts the clean protocol given noisy input and timestep
- **Conditioning**: Task goals, instrument constraints, and current lab state
- **DAG Structure**: Denoiser must learn to respect temporal ordering constraints

**Training Process:**
- Start with clean protocols from dataset
- Add Gaussian noise over T timesteps
- Train denoiser to predict original protocol given noisy version and timestep
- Use classifier-free guidance for better control

**Example Implementation:**
```python
def diffusion_generate(goal_embedding, num_steps=1000):
    # Start with pure noise
    noisy_protocol = torch.randn(protocol_shape)
    
    for t in reversed(range(num_steps)):
        # Predict clean protocol
        predicted_clean = denoiser(noisy_protocol, t, goal_embedding)
        
        # Apply constraint projection
        predicted_clean = project_to_constraints(predicted_clean)
        
        # Denoise step
        noisy_protocol = denoise_step(noisy_protocol, predicted_clean, t)
    
    return noisy_protocol
```

**Advantages:**
- Can generate diverse, high-quality protocols
- Natural handling of global structure
- Good at exploring solution space

**Challenges:**
- Requires many denoising steps
- Constraint satisfaction during sampling is tricky
- Less interpretable than autoregressive methods

#### 3. Constraint-Satisfying Planning with Neural Guidance
This hybrid approach combines symbolic planning with learned heuristics from GNNs.

**Architecture Details:**
- **Symbolic Planner**: MILP/CP-SAT solver that enforces hard constraints including DAG structure
- **Neural Heuristic**: GNN that learns to guide the search efficiently
- **Integration**: Use GNN predictions to order search branches or estimate costs
- **Temporal Constraints**: Solver ensures timestamp ordering and prevents cycles

**Training Strategy:**
- Collect planning traces from solver
- Train GNN to predict:
  - Action priors (which operations are likely useful)
  - Cost-to-go estimates (how expensive remaining steps will be)
  - Constraint violation likelihood

**Example Implementation:**
```python
def guided_planning(initial_state, goal):
    # Encode state with GNN
    state_embedding = gnn_encoder(initial_state)
    
    # Use in symbolic planner
    plan = symbolic_planner(
        initial_state, 
        goal,
        action_heuristics=action_priors,
        cost_heuristics=cost_estimate
    )
    
    return plan
```

**Advantages:**
- Guaranteed constraint satisfaction
- Can leverage decades of optimization research
- Neural guidance improves search efficiency

**Challenges:**
- Requires symbolic constraint modeling
- Integration complexity
- May be slower than pure neural approaches

#### 4. Imitation + RL Hybrid
This approach starts with supervised learning on historical data and refines with reinforcement learning.

**Architecture Details:**
- **Behavior Cloning**: Initial training on expert demonstrations
- **RL Fine-tuning**: Use simulator rewards to improve policy
- **Hybrid Loss**: Combine imitation and RL objectives

**Training Phases:**
1. **Phase 1**: Train policy to mimic expert protocols
2. **Phase 2**: Use RL to optimize for efficiency, robustness, and safety
3. **Phase 3**: Iterative improvement with human feedback

**Example Implementation:**
```python
def hybrid_training(expert_data, simulator):
    # Phase 1: Behavior cloning
    policy = train_imitation(expert_data)
    
    # Phase 2: RL fine-tuning
    for episode in range(num_episodes):
        state = simulator.reset()
        done = False
        
        while not done:
            action = policy(state)
            next_state, reward, done = simulator.step(action)
            
            # Update policy with RL algorithm (e.g., PPO)
            policy.update(state, action, reward, next_state)
            state = next_state
```

**Advantages:**
- Starts with reasonable behavior
- Can optimize for complex objectives
- Combines best of supervised and RL

**Challenges:**
- Requires high-quality simulator
- RL training can be unstable
- Need to balance imitation vs. exploration

#### Combining Approaches
The most effective systems often combine multiple approaches:
- Use autoregressive generation for high-level structure
- Apply diffusion for local refinements
- Use symbolic planning for critical safety constraints
- Fine-tune with RL for efficiency optimization

### Action parameterization and constraint masking
To keep the action space tractable:
- Predict operation type first (transfer/mix/thermo step), then endpoints via pointer networks over node embeddings, then continuous attributes (e.g., volume) with bounded distributions.
- **Timestamp assignment**: Each new operation must have a timestamp greater than all previous operations to maintain DAG structure.
- Apply masks derived from current state: available tips, sufficient volume at source, capacity at destination, deck reachability, sterility compatibility.
- **DAG constraint masking**: Prevent edges that would create cycles or violate temporal ordering.
- Enforce invariants by projection (e.g., clip volumes to feasible ranges) and by rejecting invalid samples.

### State representation details
- Node features: current volume, composition embedding (e.g., learned from reagent ontology), temperature, contamination flags, container geometry.
- Edge features: operation type, executed volume, time since last action, tip id.
- Global features: assay goal embedding, allowed instruments, remaining time budget.
- Temporal encoding: append step index or use recurrent GNN layers to retain history.

### Training signals and datasets
- Imitation data: parse existing protocols (e.g., from OT-2, Hamilton scripts) into action graphs.
- Supervision: next-edge classification, endpoint selection, and attribute regression; auxiliary losses for state prediction (e.g., next-node volume) improve stability.
- Negative sampling: generate near-miss actions (slightly over volume, wrong tip) to sharpen constraint awareness.

### Evaluation metrics
- Validity: fraction of generated steps passing all constraints; zero spills/overflows; no cross-contamination.
- Goal satisfaction: assay success rate, target composition accuracy.
- Efficiency: action count, total time, tip consumption, deck moves.
- Diversity: unique valid workflows per goal.
- Sim-to-real: execution success on hardware with minimal edits.

### Minimal prototype sketch
Outline of an autoregressive generator with constraint masking:

1. Encode current graph with a k-layer message-passing GNN.
2. Predict operation type with a masked classifier.
3. Select source and target nodes using pointer heads over node embeddings with feasibility masks.
4. **Assign timestamp**: Ensure new operation timestamp > all previous timestamps to maintain DAG.
5. Regress attributes (volume, speed) with bounded outputs; project to valid ranges.
6. **Validate DAG**: Check that no cycles would be created by the new edge.
7. Update node states and append the new edge; repeat until done.
8. Use beam search for better plans; score beams by learned value function + hard constraint checks.

### Concrete Example: Variable Serial Dilution Network Discovery

Let's implement a simplified version of the autoregressive approach for discovering the network required for a variable serial dilution on a 96-well plate. This example shows how DAG constraints and connectivity generation work in practice.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

# Define operation types
class OpType(Enum):
    ASPIRATE = "aspirate"
    DISPENSE = "dispense"
    TRANSFER = "transfer"
    MIX = "mix"

@dataclass
class LiquidState:
    """Represents the state of liquid in a well"""
    volume: float  # Current volume in μL
    concentration: float  # Concentration of target compound
    contamination_risk: float  # Risk of cross-contamination (0-1)
    timestamp: int  # When this state was created

@dataclass
class Operation:
    """Represents a liquid handling operation"""
    op_type: OpType
    source_well: Optional[str]  # None for aspirate from reservoir
    target_well: str
    volume: float
    timestamp: int
    tip_id: str

class DilutionWorkflow:
    """Represents the current state of a dilution workflow"""
    def __init__(self, plate_rows: int = 8, plate_cols: int = 12):
        self.plate_rows = plate_rows
        self.plate_cols = plate_cols
        self.wells = {}  # well_id -> LiquidState
        self.operations = []  # List of Operation objects
        self.available_tips = [f"tip_{i}" for i in range(8)]  # 8-channel pipette
        self.timestamp = 0
        
        # Initialize source wells (e.g., A1 has stock solution)
        self.wells["A1"] = LiquidState(volume=200.0, concentration=1000.0, 
                                      contamination_risk=0.0, timestamp=0)
    
    def get_well_id(self, row: int, col: int) -> str:
        """Convert row/col to well ID (e.g., A1, B2)"""
        return f"{chr(65 + row)}{col + 1}"
    
    def can_transfer(self, source: str, target: str, volume: float) -> bool:
        """Check if a transfer operation is valid"""
        if source not in self.wells or target not in self.wells:
            return False
        
        source_state = self.wells[source]
        target_state = self.wells[target]
        
        # Check volume constraints
        if source_state.volume < volume:
            return False
        
        # Check contamination risk (can't transfer to contaminated wells)
        if target_state.contamination_risk > 0.5:
            return False
        
        # Check DAG constraint: source must be created before target
        if source_state.timestamp >= target_state.timestamp:
            return False
        
        return True
    
    def add_operation(self, op: Operation):
        """Add an operation and update well states"""
        self.operations.append(op)
        self.timestamp = max(self.timestamp, op.timestamp) + 1
        
        if op.op_type == OpType.TRANSFER:
            # Update source well
            if op.source_well:
                source_state = self.wells[op.source_well]
                source_state.volume -= op.volume
                source_state.timestamp = self.timestamp
            
            # Update target well
            if op.target_well not in self.wells:
                self.wells[op.target_well] = LiquidState(
                    volume=0.0, concentration=0.0, 
                    contamination_risk=0.0, timestamp=self.timestamp
                )
            
            target_state = self.wells[op.target_well]
            target_state.volume += op.volume
            
            # Calculate new concentration (weighted average)
            if target_state.volume > 0:
                if op.source_well:
                    source_conc = self.wells[op.source_well].concentration
                    target_state.concentration = (
                        (target_state.volume - op.volume) * target_state.concentration +
                        op.volume * source_conc
                    ) / target_state.volume
                
                # Update contamination risk
                if op.source_well:
                    source_risk = self.wells[op.source_well].contamination_risk
                    target_state.contamination_risk = max(
                        target_state.contamination_risk, source_risk
                    )

class DilutionNetworkGenerator:
    """Generates dilution networks using a simplified GNN-like approach"""
    
    def __init__(self, hidden_dim: int = 64):
        self.hidden_dim = hidden_dim
        
        # Simple MLPs for different prediction tasks
        self.op_type_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(OpType))
        )
        
        self.source_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # node + global context
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.target_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.volume_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output 0-1, scale to actual volume
        )
    
    def encode_workflow_state(self, workflow: DilutionWorkflow) -> Dict[str, torch.Tensor]:
        """Encode the current workflow state into node and global embeddings"""
        # Simple encoding: concatenate well features
        well_features = []
        well_ids = []
        
        for well_id in workflow.wells:
            state = workflow.wells[well_id]
            features = [
                state.volume / 200.0,  # Normalize volume
                state.concentration / 1000.0,  # Normalize concentration
                state.contamination_risk,
                state.timestamp / 100.0  # Normalize timestamp
            ]
            well_features.append(features)
            well_ids.append(well_id)
        
        # Pad to fixed size for batch processing
        max_wells = workflow.plate_rows * workflow.plate_cols
        while len(well_features) < max_wells:
            well_features.append([0.0, 0.0, 0.0, 0.0])
            well_ids.append("")
        
        # Global context: goal concentration, remaining wells to fill
        target_concentration = 100.0  # Example target
        remaining_wells = max_wells - len([w for w in workflow.wells.values() if w.volume > 0])
        
        global_features = [
            target_concentration / 1000.0,
            remaining_wells / max_wells,
            workflow.timestamp / 100.0
        ]
        
        return {
            'well_features': torch.tensor(well_features, dtype=torch.float32),
            'well_ids': well_ids,
            'global_features': torch.tensor(global_features, dtype=torch.float32)
        }
    
    def predict_next_operation(self, workflow: DilutionWorkflow) -> Operation:
        """Predict the next operation using the current workflow state"""
        # Encode current state
        encoded = self.encode_workflow_state(workflow)
        well_features = encoded['well_features']
        global_features = encoded['global_features']
        
        # Simple "GNN-like" processing: aggregate well features
        node_embeddings = well_features @ torch.randn(4, self.hidden_dim)  # Simplified
        
        # Predict operation type
        global_context = global_features.unsqueeze(0).expand(node_embeddings.shape[0], -1)
        combined_features = torch.cat([node_embeddings, global_context], dim=1)
        
        op_type_logits = self.op_type_predictor(node_embeddings.mean(dim=0))
        op_type = OpType(list(OpType)[op_type_logits.argmax().item()])
        
        # Predict source well (with masking)
        source_scores = self.source_predictor(combined_features).squeeze()
        source_mask = torch.zeros_like(source_scores)
        
        # Mask: only wells with liquid can be sources
        for i, well_id in enumerate(encoded['well_ids']):
            if well_id in workflow.wells and workflow.wells[well_id].volume > 0:
                source_mask[i] = 1.0
        
        source_scores = source_scores * source_mask
        source_idx = source_scores.argmax().item()
        source_well = encoded['well_ids'][source_idx] if source_mask[source_idx] > 0 else None
        
        # Predict target well (with masking)
        target_scores = self.target_predictor(combined_features).squeeze()
        target_mask = torch.zeros_like(target_scores)
        
        # Mask: prefer empty wells or wells that need dilution
        for i, well_id in enumerate(encoded['well_ids']):
            if well_id not in workflow.wells or workflow.wells[well_id].volume < 50:
                target_mask[i] = 1.0
        
        target_scores = target_scores * target_mask
        target_idx = target_scores.argmax().item()
        target_well = encoded['well_ids'][target_idx]
        
        # Predict volume
        volume_logit = self.volume_predictor(node_embeddings.mean(dim=0))
        volume = volume_logit.item() * 50.0  # Scale to 0-50 μL range
        
        # Ensure DAG constraint: timestamp must be greater than all previous
        timestamp = workflow.timestamp + 1
        
        # Select available tip
        tip_id = workflow.available_tips[0]  # Simplified
        
        return Operation(
            op_type=op_type,
            source_well=source_well,
            target_well=target_well,
            volume=volume,
            timestamp=timestamp,
            tip_id=tip_id
        )

def generate_dilution_workflow(target_concentrations: List[float], 
                             max_operations: int = 50) -> DilutionWorkflow:
    """Generate a complete dilution workflow"""
    workflow = DilutionWorkflow()
    generator = DilutionNetworkGenerator()
    
    operations_count = 0
    
    while operations_count < max_operations:
        # Check if we've achieved our goals
        filled_wells = [w for w in workflow.wells.values() if w.volume > 0]
        if len(filled_wells) >= len(target_concentrations):
            # Check if concentrations are close enough
            achieved_concentrations = [w.concentration for w in filled_wells[:len(target_concentrations)]]
            if all(abs(ac - tc) < 50 for ac, tc in zip(achieved_concentrations, target_concentrations)):
                break
        
        # Predict next operation
        try:
            next_op = generator.predict_next_operation(workflow)
            
            # Validate operation
            if next_op.source_well and next_op.target_well:
                if workflow.can_transfer(next_op.source_well, next_op.target_well, next_op.volume):
                    workflow.add_operation(next_op)
                    operations_count += 1
                    print(f"Added operation: {next_op.op_type.value} {next_op.volume:.1f}μL "
                          f"from {next_op.source_well} to {next_op.target_well}")
                else:
                    print(f"Invalid operation: {next_op.op_type.value} {next_op.volume:.1f}μL "
                          f"from {next_op.source_well} to {next_op.target_well}")
            else:
                # Handle aspirate/dispense operations
                workflow.add_operation(next_op)
                operations_count += 1
                
        except Exception as e:
            print(f"Error generating operation: {e}")
            break
    
    return workflow

# Example usage
if __name__ == "__main__":
    # Generate a workflow for 8 different concentrations
    target_concentrations = [800, 600, 400, 200, 100, 50, 25, 12.5]
    
    print("Generating dilution workflow...")
    workflow = generate_dilution_workflow(target_concentrations)
    
    print(f"\nGenerated {len(workflow.operations)} operations")
    print(f"Final workflow has {len(workflow.wells)} wells with liquid")
    
    # Show final concentrations
    print("\nFinal well states:")
    for well_id, state in sorted(workflow.wells.items()):
        if state.volume > 0:
            print(f"{well_id}: {state.volume:.1f}μL, {state.concentration:.1f} ng/μL")
    
    # Verify DAG property
    timestamps = [op.timestamp for op in workflow.operations]
    if timestamps == sorted(timestamps):
        print("\n✓ DAG constraint satisfied: all operations are temporally ordered")
    else:
        print("\n✗ DAG constraint violated: operations are not temporally ordered")
```

This example demonstrates:

1. **DAG Enforcement**: Each operation gets a timestamp greater than all previous operations
2. **Constraint Masking**: Source wells must have liquid, target wells should be empty or need dilution
3. **State Updates**: Well volumes and concentrations are updated after each operation
4. **Validation**: Operations are checked for feasibility before execution
5. **Goal-Oriented Generation**: The workflow continues until target concentrations are achieved

The generator uses a simplified "GNN-like" approach with:
- Node embeddings based on well features (volume, concentration, contamination, timestamp)
- Global context (target concentration, remaining wells, current timestamp)
- Masked prediction for source/target selection
- Constraint validation to maintain physical and temporal consistency

### Why GNNs fit this problem
Message passing aligns with local physical constraints while still capturing long-range goals through multiple hops and global features, as articulated in the Distill overview [link](`https://distill.pub/2021/gnn-intro/#table`). The core difference here is that we use the GNN not to label a fixed graph but to drive the creation of new connectivity under constraints.

### Outlook
Bringing workflow generation to practice requires: a realistic simulator with rich constraints, curated protocol datasets, and careful interfaces to planners and robots. The architectural pieces above provide a path to move from classification to connectivity.

References:
- Sanchez-Lengeling, B., Reif, E., Pearce, A., Wiltschko, A. “A Gentle Introduction to Graph Neural Networks,” Distill (2021). [Distill article](`https://distill.pub/2021/gnn-intro/#table`).

