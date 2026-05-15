namespace DazzSem;

public record AttributeDef(string Name, int NumValues, bool IsInput);
    
public record Observation(double[][] InputValues, double[][] OutputValues);

public class FdtNode
{
    public int Id { get; set; }
    public int ParentId { get; set; } = -1;
    public string ChosenAttribute { get; set; } = "-";
    public int AttributeValue { get; set; } = 0;
    public double Entropy { get; set; }
    public double[] ClassProbabilities { get; set; } = [];
    public double BranchFrequency { get; set; }
    public double ConditionedEntropy { get; set; }
    public double InformationValue { get; set; }
    public List<FdtNode> Children { get; set; } = [];
}

public class FuzzyDecisionTree
{
    private readonly List<AttributeDef> _inputs;
    private readonly List<AttributeDef> _outputs;
    private readonly double _alpha;
    private readonly double _beta;
    private int _nodeCounter = 0;

    public FuzzyDecisionTree(List<AttributeDef> inputs, List<AttributeDef> outputs, double alpha, double beta)
    {
        _inputs = inputs;
        _outputs = outputs;
        _alpha = alpha;
        _beta = beta;
    }

    public FdtNode BuildTree(List<Observation> obs)
    {
        int numObs = obs.Count;
        var initialWeights = Enumerable.Repeat(1.0, numObs).ToArray();
        var availableAttrIndices = Enumerable.Range(0, _inputs.Count).ToList();
        
        return BuildNode(obs, initialWeights, availableAttrIndices, -1, numObs, 0);
    }

    private FdtNode BuildNode(List<Observation> obs, double[] weights, List<int> attrIdxs, int pid, double totalN, int valN)
    {
        int id = ++_nodeCounter;
        double mass = weights.Sum();
        double freq = mass / totalN;

        // Compute class probabilities
        int numClasses = _outputs[0].NumValues;
        var classMasses = Enumerable.Range(0, numClasses).Select(b => obs.Select((o, k) => weights[k] * o.OutputValues[0][b]).Sum()).ToArray();
        var probs = classMasses.Select(m => mass > 0 ? m / mass : 0.0).ToArray();
        double nodeEntropy = probs.Where(p => p > 0).Select(p => -p * Math.Log2(p)).Sum() * totalN * freq;

        if (freq < _alpha || probs.Any(p => p >= _beta) || attrIdxs.Count == 0)
            return new FdtNode {
                Id = id,
                ParentId = pid,
                AttributeValue = valN,
                Entropy = nodeEntropy,
                ClassProbabilities = probs,
                BranchFrequency = freq};

        // Find best attribute (min conditioned entropy)
        int bestAttr = -1;
        double minCondEntropy = double.MaxValue;

        foreach (var idx in attrIdxs)
        {
            double condEntropy = 0;
            for (int j = 0; j < _inputs[idx].NumValues; j++)
            {
                double subMass = obs.Select((o, k) => weights[k] * o.InputValues[idx][j]).Sum();
                if (subMass <= 0) continue;

                double subEntropy = 0;
                for (int b = 0; b < numClasses; b++)
                {
                    double joint = obs.Select((o, k) => weights[k] * o.InputValues[idx][j] * o.OutputValues[0][b]).Sum();
                    if (joint > 0) { double p = joint / subMass; subEntropy -= p * Math.Log2(p); }
                }
                condEntropy += subMass * subEntropy;
            }

            if (condEntropy < minCondEntropy) { minCondEntropy = condEntropy; bestAttr = idx; }
        }

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

        // Branching
        var remaining = attrIdxs.Where(i => i != bestAttr).ToList();
        for (int j = 0; j < _inputs[bestAttr].NumValues; j++)
        {
            var nextWeights = weights.Select((w, k) => w * obs[k].InputValues[bestAttr][j]).ToArray();
            node.Children.Add(BuildNode(obs, nextWeights, remaining, id, totalN, j + 1));
        }

        return node;
    }

    // --- Prediction ---
    public List<double[]> Predict(FdtNode root, List<double[][]> queries)
    {
        return [.. queries.Select(q => {
                var res = new double[_outputs[0].NumValues];
                Walk(root, q, 1.0, res);
                double sum = res.Sum();
                return sum > 0 ? res.Select(v => v / sum).ToArray() : res;
            })];
    }
    
    private void Walk(FdtNode node, double[][] q, double w, double[] res)
    {
        if (node.ChosenAttribute == "-")
        {
            for (int i = 0; i < res.Length; i++) res[i] += w * node.ClassProbabilities[i];
            return;
        }
        int idx = _inputs.FindIndex(a => a.Name == node.ChosenAttribute);
        for (int j = 0; j < node.Children.Count; j++)
        {
            double nw = w * q[idx][j];
            if (nw > 0) Walk(node.Children[j], q, nw, res);
        }
    }
}