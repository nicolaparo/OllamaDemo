using OllamaSharp;

namespace OllamaDemo
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            //await ChatDemoAsync();
            //await EmbeddingsDemoAsync();
            await VisionDemoAsync();
        }

        private static async Task ChatDemoAsync()
        {
            var client = new OllamaApiClient("http://localhost:11434", "gemma3:1b");

            var chat = new Chat(client, "You are a helpful assistant. Answer the user behaving like a pirate.");

            while (true)
            {
                var input = Console.ReadLine();
                if (input == "/exit")
                    break;

                Console.ForegroundColor = ConsoleColor.Blue;
                await chat.SendAsync(input).StreamToEndAsync(Console.Write);
                Console.ResetColor();

                Console.WriteLine(); // Add a new line for better readability
            }
        }

        private static async Task EmbeddingsDemoAsync()
        {
            var client = new OllamaApiClient("http://localhost:11434", "nomic-embed-text");

            var knowledge = new List<string>
            {
                "Tomatoes are a fruit.",
                "Apples are also a fruit.",
                "Bananas are a fruit too.",
                "Oranges are citrus fruits."
            };

            var knowledgeEmbeddings = (await client.EmbedAsync(new OllamaSharp.Models.EmbedRequest { Input = knowledge })).Embeddings;

            var input = "Tangerine are citrus fruits as well";
            var inputEmbeddings = (await client.EmbedAsync(input)).Embeddings[0];

            // Calculate cosine distance between input and knowledge embeddings

            Console.WriteLine($"Input: {input}");

            foreach (var (embedding, index) in knowledgeEmbeddings.Select((e, i) => (e, i)))
            {
                var similarity = CosineSimilarity(inputEmbeddings, embedding);
                Console.WriteLine($"Distance to '{knowledge[index]}': {similarity}");
            }
        }

        private static float CosineSimilarity(float[] a, float[] b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("Vectors must be of the same length.");

            float dotProduct = 0;
            float normA = 0;
            float normB = 0;
            for (int i = 0; i < a.Length; i++)
            {
                dotProduct += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }
            if (normA == 0 || normB == 0)
                return float.MaxValue;

            return (dotProduct / (MathF.Sqrt(normA) * MathF.Sqrt(normB)));
        }

        private static async Task VisionDemoAsync()
        {
            var client = new OllamaApiClient("http://localhost:11434", "llava:latest");

            var chat = new Chat(client, """
                You are a smart traffic camera. you need to report the level of traffic in the city.
                You can respond with one of the following levels:
                - low
                - congested
                """);

            IEnumerable<byte>[] images = [await File.ReadAllBytesAsync("Images/empty-road.jpg")];

            var response = await chat.SendAsync("Based on the image, what is the traffic level? reply with one word only [low,congested]", images)
                .StreamToEndAsync(Console.Write);

            Console.WriteLine(response);
        }
    }
}
