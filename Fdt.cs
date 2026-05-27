namespace DazzSem;

/// <summary>
/// Represents the definition of an attribute, containing its name, 
/// the number of linguistic/fuzzy values it can take, and whether it is an input attribute.
/// </summary>
public record AttributeDef(string Name, int NumValues, bool IsInput);
    
/// <summary>
/// Holds a single observation row where values are expressed as fuzzy membership degrees 
/// across all linguistic terms for input attributes and output attributes.
/// </summary>
public record Observation(double[][] InputValues, double[][] OutputValues);

/// <summary>
/// Represents a node within the Fuzzy Decision Tree. 
/// Serves as either an internal structural split or a leaf classification node.
/// </summary>
public class FdtNode
{
    public int Id { get; set; }
    public int ParentId { get; set; } = -1;
    public string ChosenAttribute { get; set; } = "-"; // If "-", this node functions as a leaf node
    public int AttributeValue { get; set; } = 0;        // The specific branch index (1-based) chosen from parent attribute split
    public double Entropy { get; set; }                  // Scaled cumulative node entropy H(S)
    public double[] ClassProbabilities { get; set; } = []; // Calculated distribution across target classes at this node
    public double BranchFrequency { get; set; }         // Relative sample mass reaching this node (mass / totalN)
    public double ConditionedEntropy { get; set; }      // Combined entropy across child branch outcomes
    public double InformationValue { get; set; }       // Information Gain computed for the chosen split
    public List<FdtNode> Children { get; set; } = [];   // Links to sub-nodes stemming from linguistic variations
}

/// <summary>
/// Implements an Un-ordered Fuzzy Decision Tree (FDT) building engine and classifier 
/// powered by cumulative information estimations and pre-pruning thresholds.
/// </summary>
public class FuzzyDecisionTree
{
    private readonly List<AttributeDef> _inputs;
    private readonly List<AttributeDef> _outputs;
    private readonly double _alpha; // Minimum branch frequency threshold below which branching is pruned
    private readonly double _beta;  // Class probability dominance threshold to safely flag a leaf node
    private int _nodeCounter = 0;

    public FuzzyDecisionTree(List<AttributeDef> inputs, List<AttributeDef> outputs, double alpha, double beta)
    {
        _inputs = inputs;
        _outputs = outputs;
        _alpha = alpha;
        _beta = beta;
    }

    /// <summary>
    /// Entry point to assemble the Fuzzy Decision Tree from the parsed training observations.
    /// </summary>
    public FdtNode BuildTree(List<Observation> obs)
    {
        int numObs = obs.Count;
        
        // At the root node, every observation initially carries a full membership weight of 1.0.
        var initialWeights = Enumerable.Repeat(1.0, numObs).ToArray();
        
        // Track the indices of input attributes that remain available for splitting down the tree.
        var availableAttrIndices = Enumerable.Range(0, _inputs.Count).ToList();
        
        // Kick off recursive tree assembly starting from the root.
        return BuildNode(obs, initialWeights, availableAttrIndices, -1, numObs, 0);
    }

    /// <summary>
    /// Recursively designs individual tree nodes by analyzing relative fuzzy subset masses,
    /// enforcing pre-pruning benchmarks, and picking optimal splitting attributes via Shannon Entropy minimization.
    /// </summary>
    private FdtNode BuildNode(List<Observation> obs, double[] weights, List<int> attrIdxs, int pid, double totalN, int valN)
    {
        int id = ++_nodeCounter;
        
        // Calculate the cumulative node mass—the sum of fractional memberships tracking down to this path.
        double mass = weights.Sum();
        
        // Branch Frequency represents how prevalent this branch is compared to the complete initial data block.
        double freq = mass / totalN;

        // --- Step 1: Compute Class Probabilities ---
        // Accumulate membership subsets for each target classification outcome mapped against current branch weights.
        int numClasses = _outputs[0].NumValues;
        var classMasses = Enumerable.Range(0, numClasses).Select(b => obs.Select((o, k) => weights[k] * o.OutputValues[0][b]).Sum()).ToArray();
        
        // Normalize class metrics into a coherent probability distribution vector.
        var probs = classMasses.Select(m => mass > 0 ? m / mass : 0.0).ToArray();
        
        // Compute standard node Shannon entropy: H(S) = -Sum( p * log2(p) ). 
        // This value is scaled to represent absolute local information content.
        double nodeEntropy = probs.Where(p => p > 0).Select(p => -p * Math.Log2(p)).Sum() * totalN * freq;

        // --- Step 2: Enforce Pre-Pruning Criteria ---
        // Stop splitting and finalize as a Leaf Node if:
        // A) The node frequency falls under our alpha floor (not enough statistical mass).
        // B) A single target classification probability breaks past our beta threshold (confident class matching).
        // C) We run out of unused features/attributes to evaluate.
        if (freq < _alpha || probs.Any(p => p >= _beta) || attrIdxs.Count == 0)
            return new FdtNode {
                Id = id,
                ParentId = pid,
                AttributeValue = valN,
                Entropy = nodeEntropy,
                ClassProbabilities = probs,
                BranchFrequency = freq};

        // --- Step 3: Optimize Attribute Selection ---
        // Evaluate remaining attributes to discover the feature that minimizes conditioned (residual) entropy.
        int bestAttr = -1;
        double minCondEntropy = double.MaxValue;

        foreach (var idx in attrIdxs)
        {
            double condEntropy = 0;
            
            // Loop over each potential linguistic value of the candidate attribute
            for (int j = 0; j < _inputs[idx].NumValues; j++)
            {
                // Find total cumulative mass allocated to this branch variation
                double subMass = obs.Select((o, k) => weights[k] * o.InputValues[idx][j]).Sum();
                if (subMass <= 0) continue;

                // Determine Shannon distribution values restricted inside this specific linguistic window
                double subEntropy = 0;
                for (int b = 0; b < numClasses; b++)
                {
                    double joint = obs.Select((o, k) => weights[k] * o.InputValues[idx][j] * o.OutputValues[0][b]).Sum();
                    if (joint > 0) { double p = joint / subMass; subEntropy -= p * Math.Log2(p); }
                }
                
                // Weight the resulting sub-entropy by absolute branch coverage mass
                condEntropy += subMass * subEntropy;
            }

            // Capture the attribute that produces the lowest overall post-split uncertainty (highest Info Gain)
            if (condEntropy < minCondEntropy) { minCondEntropy = condEntropy; bestAttr = idx; }
        }

        // --- Step 4: Construct Structural Node ---
        var node = new FdtNode
        {
            Id = id,
            ParentId = pid,
            AttributeValue = valN,
            ChosenAttribute = _inputs[bestAttr].Name,
            Entropy = nodeEntropy,
            ClassProbabilities = probs,
            BranchFrequency = freq,
            ConditionedEntropy = minCondEntropy,
            InformationValue = nodeEntropy - minCondEntropy
        };

        // --- Step 5: Branching Generation ---
        // Filter out the selected splitting feature to prevent redundant cyclical checks down the line.
        var remaining = attrIdxs.Where(i => i != bestAttr).ToList();
        
        // Spawn independent child nodes mapped to each linguistic setting of the chosen attribute.
        for (int j = 0; j < _inputs[bestAttr].NumValues; j++)
        {
            // Compute intersection weight updates via fuzzy conjunction (multiplication of memberships).
            var nextWeights = weights.Select((w, k) => w * obs[k].InputValues[bestAttr][j]).ToArray();
            
            // Recurse to generate child nodes
            node.Children.Add(BuildNode(obs, nextWeights, remaining, id, totalN, j + 1));
        }

        return node;
    }

    /// <summary>
    /// Processes incoming inference vectors by walking down structural rules 
    /// and returning consolidated class distribution predictions.
    /// </summary>
    public List<double[]> Predict(FdtNode root, List<double[][]> queries)
    {
        return [.. queries.Select(q => {
                var res = new double[_outputs[0].NumValues];
                
                // Traverse through the structural rules starting at the root node.
                Walk(root, q, 1.0, res);
                
                // Normalize the distributed weight totals to sum to 1.0 across class labels.
                double sum = res.Sum();
                return sum > 0 ? res.Select(v => v / sum).ToArray() : res;
            })];
    }
    
    /// <summary>
    /// Executes parallel fuzzy model exploration. Since items carry fractional membership 
    /// profiles, a single sample may navigate down multiple branches concurrently.
    /// </summary>
    private void Walk(FdtNode node, double[][] q, double w, double[] res)
    {
        // If we land on a leaf node, accumulate our scale-weighted distribution scores.
        if (node.ChosenAttribute == "-")
        {
            for (int i = 0; i < res.Length; i++) res[i] += w * node.ClassProbabilities[i];
            return;
        }
        
        // Identify input feature position mapped to the split parameter of this node.
        int idx = _inputs.FindIndex(a => a.Name == node.ChosenAttribute);
        
        // Concurrently traverse down all valid structural children branches.
        for (int j = 0; j < node.Children.Count; j++)
        {
            // Update tracking weights through cumulative conditional intersection logic.
            double nw = w * q[idx][j];
            
            // Only continue tracking down paths that yield an active membership signal.
            if (nw > 0) Walk(node.Children[j], q, nw, res);
        }
    }
}