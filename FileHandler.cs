namespace DazzSem;

public static class FileHandler
{
    public static (List<AttributeDef> inputs, List<AttributeDef> outputs, List<Observation> obs, double alpha, double beta, List<double[][]> queries) ParseCsv(string path)
    {
        var lines = File.ReadAllLines(path).Where(l => !string.IsNullOrWhiteSpace(l)).ToArray();
        int lineIdx = 0;

        // Line 1: num_input; num_output
        var counts = lines[lineIdx++].Split(';');
        int numInputs = int.Parse(counts[0]);
        int numOutputs = int.Parse(counts[1]); // Assuming 1 based on FDT context

        // Line 2: input attributes (name:num)
        var inputs = ParseAttributes(lines[lineIdx++], true);
        
        // Line 3: output attributes
        var outputs = ParseAttributes(lines[lineIdx++], false);

        // Line 4: num observations
        int numObs = int.Parse(lines[lineIdx++].Trim());

        // Lines 5...: observations
        var observations = new List<Observation>();
        for (int i = 0; i < numObs; i++)
        {
            var vals = lines[lineIdx++].Split(';').Select(v => double.Parse(v)).ToArray();
            observations.Add(ParseObservationRow(vals, inputs, outputs));
        }

        // Next line: alpha; beta
        var thresholds = lines[lineIdx++].Split(';');
        double alpha = double.Parse(thresholds[0]);
        double beta = double.Parse(thresholds[1]);

        // Remaining lines: queries
        var queries = new List<double[][]>();
        while (lineIdx < lines.Length)
        {
            var vals = lines[lineIdx++].Split(';').Select(v => double.Parse(v)).ToArray();
            queries.Add(ParseObservationRow(vals, inputs, new List<AttributeDef>()).InputValues);
        }

        return (inputs, outputs, observations, alpha, beta, queries);
    }

    private static List<AttributeDef> ParseAttributes(string line, bool isInput)
    {
        return [.. line.Split(';', StringSplitOptions.RemoveEmptyEntries)
                    .Select(part => part.Split(':'))
                    .Select(parts => new AttributeDef(parts[0].Trim(), int.Parse(parts[1]), isInput))];
    }

    private static Observation ParseObservationRow(double[] vals, List<AttributeDef> inputs, List<AttributeDef> outputs)
    {
        int pointer = 0;
        var inVals = new double[inputs.Count][];
        for (int i = 0; i < inputs.Count; i++)
        {
            inVals[i] = vals.Skip(pointer).Take(inputs[i].NumValues).ToArray();
            pointer += inputs[i].NumValues;
        }

        var outVals = new double[outputs.Count][];
        for (int i = 0; i < outputs.Count; i++)
        {
            outVals[i] = vals.Skip(pointer).Take(outputs[i].NumValues).ToArray();
            pointer += outputs[i].NumValues;
        }

        return new Observation(inVals, outVals);
    }

    // --- CSV Exporter ---
    public static void ExportResults(string path, FdtNode root, List<double[]> predictions, List<double[][]> queries)
    {
        using var writer = new StreamWriter(path);
        writer.WriteLine("NodeId;ParentId;Value;Attribute;H;Probabilities;Frequency;H Conditioned;I Conditioned");

        var queue = new Queue<FdtNode>();
        queue.Enqueue(root);
        while (queue.Count > 0)
        {
            var n = queue.Dequeue();
            string pStr = string.Join(", ", n.ClassProbabilities.Select(p => $"{p:F5}"));
            string attrVal = n.AttributeValue == 0 ? "-" : n.AttributeValue.ToString();
            string pid = n.ParentId > 0 ? n.ParentId.ToString() : "-";
            writer.WriteLine($"{n.Id};{pid};{attrVal};{n.ChosenAttribute};{n.Entropy:F5};[{pStr}];{n.BranchFrequency:F5};{n.ConditionedEntropy:F5};{n.InformationValue:F5}");
            foreach (var child in n.Children) queue.Enqueue(child);
        }

        writer.WriteLine();
        foreach (var (q, res) in queries.Zip(predictions))
        {
            writer.WriteLine($"{string.Join(";", q.SelectMany(x => x).Select(v => v.ToString()))};{string.Join(";", res.Select(v => v.ToString("F5")))}");
        }
    }
}