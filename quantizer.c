/**
 * Quantizer for HNN Model
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>

#define BLOCK_SIZE 256

typedef struct
{
    float scale;                       // scale
    int8_t quantized_vals[BLOCK_SIZE]; // quants with 0 zeropoint
} block_q8;

typedef struct
{
    int dim;        // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers;   // number of layers
    int n_heads;    // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len;    // max sequence length
} Config;

typedef struct
{
    float *token_embedding_table; // (vocab_size, dim)
    float *rms_att_weight;        // (layer, dim) rmsnorm weights
    float *rms_ffn_weight;        // (layer, dim)
    float *wq;                    // (layer, dim, dim)
    float *wk;                    // (layer, dim, dim)
    float *wv;                    // (layer, dim, dim)
    float *wo;                    // (layer, dim, dim)
    float *w1;                    // (layer, hidden_dim, dim)
    float *w2;                    // (layer, dim, hidden_dim)
    float *w3;                    // (layer, hidden_dim, dim)
    float *rms_final_weight;      // (dim,)
    float *freq_cis_real;         // (seq_len, dim/2)
    float *freq_cis_imag;         // (seq_len, dim/2)
    float *wcls;
} TransformerWeights;

void checkpoint_init_weights(TransformerWeights *w, Config *p, float *f, int shared_weights)
{
    int head_size = p->dim / p->n_heads;
    float *ptr = f;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += p->n_layers * p->dim;
    w->wq = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->wk = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->wv = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->wo = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += p->n_layers * p->dim;
    w->w1 = ptr;
    ptr += p->n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += p->n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += p->n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    w->freq_cis_real = ptr;
    ptr += p->seq_len * head_size / 2;
    w->freq_cis_imag = ptr;
    ptr += p->seq_len * head_size / 2;
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void get_min_max(float *ptr, int size, float *pmin, float *pmax)
{
    float min = INFINITY;
    float max = -INFINITY;

    for (int i = 0; i < size; i++)
    {
        if (ptr[i] < min)
            min = ptr[i];
        if (ptr[i] > max)
            max = ptr[i];
    }

    *pmin = min;
    *pmax = max;
}

void quantize_weights(FILE *file, float *weights, int n_layers, int layer_size, char *name)
{

    printf("------------------------\n");
    printf("Quantizing Layer [%s] : Layer_size [%d] \n", name, layer_size);

    // for each layer
    for (int l = 0; l < n_layers; l++)
    {
        // get the min and max values for this layer
        float min;
        float max;
        get_min_max(weights, layer_size, &min, &max);
        // compute the scale factor
        float scale = (max - min) / 255;
        printf("l=%d min=%f max=%f scale=%f\n", l, min, max, scale);
        // save min value and scale factor to file
        fwrite(&min, sizeof(char), 4, file);
        fwrite(&scale, sizeof(char), 4, file);
        // quantize the weights from this layer and save to file
        uint8_t qweight;
        for (int i = 0; i < layer_size; i++)
        {
            qweight = round((weights[i] - min) / (max - min) * 255);
            fwrite(&qweight, sizeof(char), 1, file);
        }
        // advance to the weights of the next layer
        weights += layer_size; // * sizeof(float);
    }
}

void quantize_block_256(float *data, block_q8 *quantized)
{
    float absmax = -INFINITY;
    for (int i = 0; i < BLOCK_SIZE; ++i)
    {
        float tmp_max = fabs(data[i]);
        if (tmp_max > absmax)
        {
            absmax = tmp_max;
        }
    }
    float scale = absmax / 127;
    quantized->scale = scale;

    for (int i = 0; i < BLOCK_SIZE; ++i)
    {
        const int8_t quantized_val = round(data[i] / scale);
        quantized->quantized_vals[i] = quantized_val;
    }
}

void quantize_weights_qi8_t(FILE *file, float *weights, int n_layers, int layer_size, char *name)
{
    printf("------------------------\n");
    printf("Quantizing Layer [%d] : Layer_size [%d] \n", name, layer_size);

    // for each layer
    for (int l = 0; l < n_layers; l++)
    {
        float scale = 0.0f;
        for (int i = 0; i < layer_size / BLOCK_SIZE; ++i)
        {
            // Move in chuncks of 256 and get the data for every chunk
            float *curr_dat = weights + i;
            block_q8 qblock;
            fwrite(&qblock.scale, sizeof(char), sizeof(float), file);
            quantize_block_256(curr_dat, &qblock);
            for (int j = 0; j < BLOCK_SIZE; ++j)
            {
                int8_t qv = qblock.quantized_vals[j];
                fwrite(&qv, sizeof(char), sizeof(int8_t), file);
            }
        }
    }
    // advance to the weights of the next layer
    weights += layer_size; // * sizeof(float);
}

void write_weights(FILE *file, float *weights, int n_layers, int layer_size, char *name)
{
    printf("------------------------\n");
    printf("Non-Quantized Layer [%d] : Layer_size [%d] \n", name, layer_size);
    char *wc = (char *)weights;
    fwrite(wc, sizeof(char), n_layers * layer_size * 4, file);
}

int convert_weights_qi8_t(TransformerWeights *w, Config *p, int shared_weights, char *output_model)
{

    FILE *file = fopen(output_model, "wb");
    if (file == NULL)
    {
        perror("Error opening file");
        return 1;
    }
    // write headers
    fwrite(&p->dim, sizeof(char), 4, file);
    fwrite(&p->hidden_dim, sizeof(char), 4, file);
    fwrite(&p->n_layers, sizeof(char), 4, file);
    fwrite(&p->n_heads, sizeof(char), 4, file);
    fwrite(&p->n_kv_heads, sizeof(char), 4, file);
    if (shared_weights)
    {
        fwrite(&p->vocab_size, sizeof(char), 4, file);
    }
    else
    {
        // this indicates unshared weights
        int neg_voca_size = -p->vocab_size;
        fwrite(&neg_voca_size, sizeof(char), 4, file);
    }

    fwrite(&p->seq_len, sizeof(char), 4, file);

    // write quantized weights
    int head_size = p->dim / p->n_heads;

    // quantize_weights(file, w->token_embedding_table, 1, p->vocab_size * p->dim, "token_embedding_table");
    write_weights(file, w->token_embedding_table, 1, p->vocab_size * p->dim, "token_embedding_table");

    // quantize_weights(file, w->rms_att_weight, p->n_layers, p->dim, "rms_att_weight");
    write_weights(file, w->rms_att_weight, p->n_layers, p->dim, "rms_att_weight");

    quantize_weights_qi8_t(file, w->wq, p->n_layers, p->dim * p->dim, "wq");
    quantize_weights_qi8_t(file, w->wk, p->n_layers, p->dim * p->dim, "wk");
    quantize_weights_qi8_t(file, w->wv, p->n_layers, p->dim * p->dim, "wv");
    // write_weights(file, w->wv ,  p->n_layers , p->dim *  p->dim);

    quantize_weights_qi8_t(file, w->wo, p->n_layers, p->dim * p->dim, "wo");

    // quantize_weights(file, w->rms_ffn_weight, p->n_layers, p->dim, "rms_ffn_weight");
    write_weights(file, w->rms_ffn_weight, p->n_layers, p->dim, "rms_ffn_weight");

    quantize_weights_qi8_t(file, w->w1, p->n_layers, p->dim * p->hidden_dim, "w1");
    quantize_weights_qi8_t(file, w->w2, p->n_layers, p->hidden_dim * p->dim, "w2");
    // write_weights(file, w->w2 ,  p->n_layers ,p->hidden_dim *  p->dim);
    quantize_weights_qi8_t(file, w->w3, p->n_layers, p->dim * p->hidden_dim, "w3");

    // quantize_weights(file, w->rms_final_weight, 1, p->dim, "rms_final_weight");
    write_weights(file, w->rms_final_weight, 1, p->dim, "rms_final_weight");
    write_weights(file, w->freq_cis_real, 1, p->seq_len * head_size / 2, "freq_cis_real");
    write_weights(file, w->freq_cis_imag, 1, p->seq_len * head_size / 2, "freq_cis_imag");
    if (!shared_weights)
    {
        // We quantize this data if top and last embedding weights are not shared
        printf("NOT Quantizing WCLS\n");
        write_weights(file, w->wcls, 1, p->vocab_size * p->dim, "wcls");
    }
    else
    {
        quantize_weights_qi8_t(file, w->wcls, 1, p->vocab_size * p->dim, "wcls");
    }
    fclose(file);
    return 0;
}

int convert_weights_qu8(TransformerWeights *w, Config *p, int shared_weights, char *output_model)
{

    FILE *file = fopen(output_model, "wb");
    if (file == NULL)
    {
        perror("Error opening file");
        return 1;
    }

    // write headers
    fwrite(&p->dim, sizeof(char), 4, file);
    fwrite(&p->hidden_dim, sizeof(char), 4, file);
    fwrite(&p->n_layers, sizeof(char), 4, file);
    fwrite(&p->n_heads, sizeof(char), 4, file);
    fwrite(&p->n_kv_heads, sizeof(char), 4, file);
    if (shared_weights)
    {
        fwrite(&p->vocab_size, sizeof(char), 4, file);
    }
    else
    {
        // this indicates unshared weights
        int neg_voca_size = -p->vocab_size;
        fwrite(&neg_voca_size, sizeof(char), 4, file);
    }

    fwrite(&p->seq_len, sizeof(char), 4, file);

    // write quantized weights
    int head_size = p->dim / p->n_heads;

    quantize_weights(file, w->token_embedding_table, 1, p->vocab_size * p->dim, "token_embedding_table");
    // write_weights(file, w->token_embedding_table, 1, p->vocab_size * p->dim, "token_embedding_table");

    // quantize_weights(file, w->rms_att_weight, p->n_layers, p->dim, "rms_att_weight");
    write_weights(file, w->rms_att_weight, p->n_layers, p->dim, "rms_att_weight");

    quantize_weights(file, w->wq, p->n_layers, p->dim * p->dim, "wq");
    quantize_weights(file, w->wk, p->n_layers, p->dim * p->dim, "wk");
    quantize_weights(file, w->wv, p->n_layers, p->dim * p->dim, "wv");
    // write_weights(file, w->wv ,  p->n_layers , p->dim *  p->dim);

    quantize_weights(file, w->wo, p->n_layers, p->dim * p->dim, "wo");

    // quantize_weights(file, w->rms_ffn_weight, p->n_layers, p->dim, "rms_ffn_weight");
    write_weights(file, w->rms_ffn_weight, p->n_layers, p->dim, "rms_ffn_weight");

    quantize_weights(file, w->w1, p->n_layers, p->dim * p->hidden_dim, "w1");
    quantize_weights(file, w->w2, p->n_layers, p->hidden_dim * p->dim, "w2");
    // write_weights(file, w->w2 ,  p->n_layers ,p->hidden_dim *  p->dim);
    quantize_weights(file, w->w3, p->n_layers, p->dim * p->hidden_dim, "w3");

    // quantize_weights(file, w->rms_final_weight, 1, p->dim, "rms_final_weight");
    write_weights(file, w->rms_final_weight, 1, p->dim, "rms_final_weight");
    write_weights(file, w->freq_cis_real, 1, p->seq_len * head_size / 2, "freq_cis_real");
    write_weights(file, w->freq_cis_imag, 1, p->seq_len * head_size / 2, "freq_cis_imag");
    if (!shared_weights)
    {
        // We quantize this data if top and last embedding weights are not shared
        printf("NOT Quantizing WCLS\n");
        write_weights(file, w->wcls, 1, p->vocab_size * p->dim, "wcls");
    }
    else
    {
        write_weights(file, w->wcls, 1, p->vocab_size * p->dim, "wcls");
    }
    fclose(file);
    return 0;
}

int main(int argc, char *argv[])
{

    int opt;
    char *input_model = NULL;
    char *output_model = NULL;
    int quantization_type = 0;

    while ((opt = getopt(argc, argv, "f:g:n:")) != -1)
    {
        switch (opt)
        {
        case 'f':
            input_model = optarg;
            break;
        case 'g':
            output_model = optarg;
            break;
        case 'n':
            quantization_type = atoi(optarg);
            break;
        default:
            fprintf(stderr, "Usage: %s -f <input_model> -g <output_model> -n <quantization_type>\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (input_model == NULL || output_model == NULL || quantization_type == -1)
    {
        fprintf(stderr, "Missing required arguments.\n");
        fprintf(stderr, "Usage: %s -f <input_model> -g <output_model> -n <quantization_type>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // read in the model.bin file
    Config config;
    TransformerWeights weights;
    int fd = 0;         // file descriptor for memory mapping
    float *data = NULL; // memory mapped data pointer
    long file_size;     // size of the input_model file in bytes
    {
        FILE *file = fopen(input_model, "rb");
        if (!file)
        {
            printf("Couldn't open file %s\n", input_model);
            return 1;
        }

        // read in the config header
        if (fread(&config, sizeof(Config), 1, file) != 1)
        {
            return 1;
        }
        // negative vocab size is hacky way of signaling unshared weights. bit yikes.
        int shared_weights = config.vocab_size > 0 ? 1 : 0;
        config.vocab_size = abs(config.vocab_size);
        // figure out the file size
        fseek(file, 0, SEEK_END); // move file pointer to end of file
        file_size = ftell(file);  // get the file size, in bytes
        fclose(file);
        printf("Original Model file size = [%ld MB]\n", file_size / 1024 / 1024);
        //  memory map the Transformer weights into the data pointer
        fd = open(input_model, O_RDONLY); // open in read only mode
        if (fd == -1)
        {
            printf("Open failed!\n");
            return 1;
        }
        data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED)
        {
            printf("mmap failed!\n");
            return 1;
        }

        // fast-forward to weight data, skipping metadata
        float *weights_ptr = data + sizeof(Config) / sizeof(float);

        // initialize all weights in float
        checkpoint_init_weights(&weights, &config, weights_ptr, shared_weights);

        int ret = 0;
        if (quantization_type == 0)
        {
            ret = convert_weights_qu8(&weights, &config, shared_weights, output_model);
        }
        else if (quantization_type == 1)
        {
            ret = convert_weights_qi8_t(&weights, &config, shared_weights, output_model);
        }
        if (ret == 0)
            printf("Model converted and saved\n");
    }

    // memory and file handles cleanup
    if (data != MAP_FAILED)
        munmap(data, file_size);
    if (fd != -1)
        close(fd);
    return 0;
}
