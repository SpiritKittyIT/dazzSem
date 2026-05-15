using DazzSem;

string inputFile = "../../../files/input.csv";
string outputFile = "../../../files/output.csv";

if (!File.Exists(inputFile))
{
    Console.WriteLine($"Error: File '{inputFile}' not found.");
    return;
}

try
{
    var (inputs, outputs, observations, alpha, beta, queries) = FileHandler.ParseCsv(inputFile);
    
    var fdt = new FuzzyDecisionTree(inputs, outputs, alpha, beta);
    var root = fdt.BuildTree(observations);
    
    var predictions = fdt.Predict(root, queries);
    
    FileHandler.ExportResults(outputFile, root, predictions, queries);
    Console.WriteLine($"Successfully processed FDT. Results saved to {outputFile}");
}
catch (Exception ex)
{
    Console.WriteLine($"An error occurred during execution: {ex.Message}");
}
