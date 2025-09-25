using Microsoft.AspNetCore.Mvc;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading.Tasks;
using System.Text.Json.Serialization;

namespace RealFakeNews.Controllers;

[ApiController]
[Route("api/ml")]
public class MLController : ControllerBase
{
    private readonly HttpClient _http;
    private readonly string _mlServiceUrl;

    private const int MaxRetries = 100;
    private const int DelayMs = 2000;

    public MLController(IHttpClientFactory httpFactory, IConfiguration config)
    {
        _http = httpFactory.CreateClient();
        _mlServiceUrl = config["ML_SERVICE_URL"] ?? "http://ml-service:8000";
    }

    private async Task<HttpResponseMessage> GetWithRetryAsync(string url)
    {
        for (int i = 0; i < MaxRetries; i++)
        {
            try
            {
                var response = await _http.GetAsync(url);
                response.EnsureSuccessStatusCode();
                return response;
            }
            catch (HttpRequestException)
            {
                if (i == MaxRetries - 1) throw;
                await Task.Delay(DelayMs);
            }
        }
        throw new HttpRequestException($"ML service {url} не відповідає після {MaxRetries} спроб");
    }

    private async Task<HttpResponseMessage> PostWithRetryAsync(string url, HttpContent? content)
    {
        for (int i = 0; i < MaxRetries; i++)
        {
            try
            {
                var response = await _http.PostAsync(url, content ?? new StringContent(""));
                response.EnsureSuccessStatusCode();
                return response;
            }
            catch (HttpRequestException)
            {
                if (i == MaxRetries - 1) throw;
                await Task.Delay(DelayMs);
            }
        }
        throw new HttpRequestException($"ML service {url} не відповідає після {MaxRetries} спроб");
    }

    // 🔹 DTO для Analyze
    public class AnalyzeRequest
    {
        public double TestSize { get; set; } = 0.3;
        public int MaxIter { get; set; } = 1000;
        public double C { get; set; } = 1.0;
        public string Solver { get; set; } = "liblinear";

        [JsonPropertyName("model_name")]
        public string ModelName { get; set; } = "logreg";
    }

    [HttpPost("preprocess")]
    public async Task<IActionResult> Preprocess([FromForm] IFormFileCollection files)
    {
        using var content = new MultipartFormDataContent();

        foreach (var file in files)
        {
            var streamContent = new StreamContent(file.OpenReadStream());
            streamContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue(file.ContentType);
            content.Add(streamContent, "files", file.FileName);
        }

        var response = await PostWithRetryAsync($"{_mlServiceUrl}/preprocess", content);
        var result = await response.Content.ReadAsStringAsync();
        return Content(result, "application/json");
    }

    [HttpPost("analyze")]
    public async Task<IActionResult> Analyze([FromBody] AnalyzeRequest request)
    {
        var response = await PostWithRetryAsync(
            $"{_mlServiceUrl}/analyze",
            JsonContent.Create(new
            {
                test_size = request.TestSize,
                max_iter = request.MaxIter,
                C = request.C,
                solver = request.Solver,
                model_name = request.ModelName
            })
        );

        var result = await response.Content.ReadAsStringAsync();
        return Content(result, "application/json");
    }

    [HttpGet("analyze/status")]
    public async Task<IActionResult> AnalyzeStatus([FromQuery] string? model_name = null)
    {
        string url = $"{_mlServiceUrl}/analyze/status";
        if (!string.IsNullOrEmpty(model_name))
        {
            url += $"?model_name={model_name}";
        }

        var response = await GetWithRetryAsync(url);
        var result = await response.Content.ReadAsStringAsync();
        return Content(result, "application/json");
    }

    public class PredictRequest
    {
        [JsonPropertyName("news_text")]
        public string NewsText { get; set; } = "";

        [JsonPropertyName("model_name")]
        public string ModelName { get; set; } = "logreg";
    }

    [HttpPost("predict")]
    public async Task<IActionResult> Predict([FromBody] PredictRequest req)
    {
        var response = await PostWithRetryAsync(
            $"{_mlServiceUrl}/predict",
            JsonContent.Create(new { news_text = req.NewsText, model_name = req.ModelName })
        );
        var result = await response.Content.ReadAsStringAsync();
        return Content(result, "application/json");
    }

    [HttpGet("random_predict")]
    public async Task<IActionResult> RandomPredict([FromQuery] string? model_name = null)
    {
        string url = $"{_mlServiceUrl}/random_predict";
        if (!string.IsNullOrEmpty(model_name))
        {
            url += $"?model_name={model_name}";
        }

        var response = await GetWithRetryAsync(url);
        var result = await response.Content.ReadAsStringAsync();
        return Content(result, "application/json");
    }

    [HttpGet("interpret/{method}")]
    public async Task<IActionResult> Interpret(string method, [FromQuery] string? model_name = null)
    {
        string url = $"{_mlServiceUrl}/interpret/{method}";
        if (!string.IsNullOrEmpty(model_name))
        {
            url += $"?model_name={model_name}";
        }

        var response = await GetWithRetryAsync(url);
        var result = await response.Content.ReadAsStringAsync();
        return Content(result, "application/json");
    }

    [HttpGet("visualize/{method}")]
    public async Task<IActionResult> Visualize(string method, [FromQuery] string? model_name = null)
    {
        string url = $"{_mlServiceUrl}/visualize/{method}";
        if (!string.IsNullOrEmpty(model_name))
        {
            url += $"?model_name={model_name}";
        }

        var response = await GetWithRetryAsync(url);
        var result = await response.Content.ReadAsStringAsync();
        return Content(result, "application/json");
    }
}
