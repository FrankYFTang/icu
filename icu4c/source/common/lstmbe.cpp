// © 2021 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#include <complex>
#include <utility>

#if !UCONFIG_NO_BREAK_ITERATION

#define CL_TARGET_OPENCL_VERSION 100
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/opencl.h>

#include "unicode/utypes.h"


#include "brkeng.h"
#include "charstr.h"
#include "cmemory.h"
#include "lstmbe.h"
#include "putilimp.h"
#include "uassert.h"
#include "ubrkimpl.h"
#include "uresimp.h"
#include "uvectr32.h"
#include "uvector.h"
#include "umutex.h"
#include "ucln_cmn.h"

#include "unicode/brkiter.h"
#include "unicode/resbund.h"
#include "unicode/ubrk.h"
#include "unicode/uniset.h"
#include "unicode/ustring.h"
#include "unicode/utf.h"
#include "unicode/uclean.h"

U_NAMESPACE_BEGIN


class OpenCL : public UMemory {
public:
    OpenCL() {
        int err;
        clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
        context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
        U_ASSERT(err == 0);

        const char *KernelSource =
  "__kernel void maxIndex(__global int* out, __global float* in, int len, const int group) { \n" \
  "   int i = get_global_id(0); \n" \
  "   if (i >= len) return; \n" \
  "   int maxIdx = 0; \n" \
  "   int offset = i * group; \n" \
  "   float max = in[offset]; \n" \
  "   for (int j = 1; j < group; j++) { \n" \
  "      if(in[offset + j] > max) { \n" \
  "        max = in[offset + j];\n" \
  "        maxIdx = j;\n" \
  "      } \n" \
  "   } \n" \
  "   out[i] = maxIdx; \n" \
  "} \n" \
  "__kernel void clear(__global float* a, const int len, const int offset) { \n" \
  "   int i = get_global_id(0); \n" \
  "   if (i >= len) return; \n" \
  "   a[i+offset] = 0; \n" \
  "} \n" \
  "__kernel void print(__global float* a, const int len, const int offset) { \n" \
  "   int i = get_global_id(0); \n" \
  "   if (i >= len) return; \n" \
  "   if (i == 0) { \n" \
  "     printf(\"\\nMAT:\"); \n" \
  "     for (int j = 0; j < len; j++) { \n" \
  "       printf(\"%f \", a[j + offset]); \n" \
  "     } \n" \
  "     printf(\"\\n\"); \n" \
  "   } \n" \
  "} \n" \
  "__kernel void ident(__global float* a, const int len, const int offset) { \n" \
  "   int i = get_global_id(0); \n" \
  "   if (i >= len) return; \n" \
  "   a[i + offset] = i; \n" \
  "} \n" \
  "__kernel void add(__global float* out, __global float* a, __global float* b, \n" \
  "                  const int len, const int offset) { \n" \
  "   int i = get_global_id(0); \n" \
  "   if (i >= len) return; \n" \
  "   out[i+offset] = a[i] + b[i];\n" \
  "} \n" \
  "__kernel void add3(__global float* out, __global float* a, __global float* b, \n" \
  "                   __global float* c, const int len) { \n" \
  "   int i = get_global_id(0); \n" \
  "   if (i >= len) return; \n" \
  "   out[i] = a[i] + b[i] + c[i];\n" \
  "} \n" \
  "__kernel void mul1d2d(__global float* out, __global float* a, __global float* b, \n" \
  "                      const int n, const int m, const int aOffset ) { \n" \
  "   int i = get_global_id(0); \n" \
  "   if (i >= n * m) return; \n" \
  "   out[i] = a[i / m  + aOffset] * b[i];\n" \
  "} \n" \
  "__kernel void offsetAdd(__global float* data, const int length, const int aOffset, const int bOffset) { \n" \
  "   int i = get_global_id(0); \n" \
  "   if (i >= length) return; \n" \
  "   data[ i + aOffset] += data[i + bOffset];\n" \
  "} \n" \
  "__kernel void tanhOrSigmoid( __global float* ifco, const int length, const int hunits) { \n" \
  "   int i = get_global_id(0); \n" \
  "   if (i >= length) return; \n" \
  "   if (i < 2 * hunits || i >= 3 * hunits) { \n" \
  "      ifco[i] = 1.0f/(1.0f + exp(-ifco[i])); \n" \
  "   } else { \n" \
  "      ifco[i] = tanh(ifco[i]); \n" \
  "   } \n" \
  "} \n" \
  "__kernel void updateCandH(__global float* ifco, __global float* c, __global float* h, \n" \
  "                       const int hunits, const int iOffset, const int fOffset, \n" \
  "                       const int cOffset, const int oOffset) { \n" \
  "   int i = get_global_id(0); \n" \
  "   if (i >= hunits) return; \n" \
  "   c[i] = c[i] * ifco[i + fOffset] + ifco[i + iOffset] * ifco[i + cOffset];\n" \
  "   h[i] = tanh(c[i]) * ifco[i + oOffset] ;\n" \
  "} \n" \
  "__kernel void copy(__global float* dest, __global float* src, \n" \
  "                       const int length, const int destOffset, const int srcOffset) { \n" \
  "   int i = get_global_id(0); \n" \
  "   if (i >= length) return; \n" \
  "   dest[i + destOffset] = src[i + srcOffset]; \n" \
  "}";

        cl_program program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
        U_ASSERT(err == 0);
        err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        U_ASSERT(err == 0);
        clear = clCreateKernel(program, "clear", &err);
        U_ASSERT(err == 0);
        ident = clCreateKernel(program, "ident", &err);
        U_ASSERT(err == 0);
        print = clCreateKernel(program, "print", &err);
        U_ASSERT(err == 0);
        maxIndex = clCreateKernel(program, "maxIndex", &err);
        U_ASSERT(err == 0);
        add = clCreateKernel(program, "add", &err);
        U_ASSERT(err == 0);
        add3 = clCreateKernel(program, "add3", &err);
        U_ASSERT(err == 0);
        mul1d2d = clCreateKernel(program, "mul1d2d", &err);
        U_ASSERT(err == 0);
        offsetAdd = clCreateKernel(program, "offsetAdd", &err);
        U_ASSERT(err == 0);
        tanhOrSigmoid = clCreateKernel(program, "tanhOrSigmoid", &err);
        U_ASSERT(err == 0);
        updateCandH = clCreateKernel(program, "updateCandH", &err);
        U_ASSERT(err == 0);
        copy = clCreateKernel(program, "copy", &err);
        U_ASSERT(err == 0);
    }
    ~OpenCL() {
        clReleaseKernel(print);
        clReleaseKernel(ident);
        clReleaseKernel(clear);
        clReleaseKernel(add3);
        clReleaseKernel(add);
        clReleaseKernel(maxIndex);
        clReleaseKernel(mul1d2d);
        clReleaseKernel(offsetAdd);
        clReleaseKernel(updateCandH);
        clReleaseKernel(tanhOrSigmoid);
        clReleaseKernel(copy);
        clReleaseContext(context);
    }
    cl_context getContext() { return context; }
    cl_device_id getDeviceID() { return device_id; }
    cl_kernel getPrint() { return print; }
    cl_kernel getIdent() { return ident; }
    cl_kernel getClear() { return clear; }
    cl_kernel getAdd() { return add; }
    cl_kernel getAdd3() { return add3; }
    cl_kernel getMaxIndex() { return maxIndex; }
    cl_kernel getMul1d2d() { return mul1d2d; }
    cl_kernel getOffsetAdd() { return offsetAdd; }
    cl_kernel getTanhOrSigmoid() { return tanhOrSigmoid; }
    cl_kernel getUpdateCH() { return updateCandH; }
    cl_kernel getCopy() { return copy; }

private:
    cl_device_id device_id;
    cl_context context;
    cl_kernel print;
    cl_kernel ident;
    cl_kernel clear;
    cl_kernel add;
    cl_kernel add3;
    cl_kernel maxIndex;
    cl_kernel mul1d2d;
    cl_kernel offsetAdd;
    cl_kernel updateCandH;
    cl_kernel tanhOrSigmoid;
    cl_kernel copy;
};

static UInitOnce gOpenCLInitOnce {};
static OpenCL* gOpenCL = nullptr;

static UBool U_CALLCONV opencl_cleanup() {
    gOpenCLInitOnce.reset();
    delete gOpenCL;
    return true;
}

static void U_CALLCONV
initOpenCL(UErrorCode&)
{
    gOpenCL = new OpenCL();
    ucln_common_registerCleanup(UCLN_COMMON_LSTM, opencl_cleanup);
}

// Uncomment the following #define to debug.
// #define LSTM_DEBUG 1
// #define LSTM_VECTORIZER_DEBUG 1

/**
 * Interface for reading 1D array.
 */
class ReadArray1D {
public:
    virtual ~ReadArray1D();
    virtual int32_t d1() const = 0;
    virtual float get(int32_t i) const = 0;
    virtual const float* getData() const = 0;
    virtual size_t dataSize() const = 0;

#ifdef LSTM_DEBUG
    void print() const {
        printf("\n[");
        for (int32_t i = 0; i < d1(); i++) {
           printf("%0.8e ", get(i));
           if (i % 4 == 3) printf("\n");
        }
        printf("]\n");
    }
#endif
};

ReadArray1D::~ReadArray1D()
{
}

/**
 * Interface for reading 2D array.
 */
class ReadArray2D {
public:
    virtual ~ReadArray2D();
    virtual int32_t d1() const = 0;
    virtual int32_t d2() const = 0;
    virtual float get(int32_t i, int32_t j) const = 0;
    virtual const float* getData() const = 0;
    virtual size_t dataSize() const = 0;
};

ReadArray2D::~ReadArray2D()
{
}

/**
 * A class to index a float array as a 1D Array without owning the pointer or
 * copy the data.
 */
class ConstArray1D : public ReadArray1D {
public:
    ConstArray1D() : data_(nullptr), d1_(0) {}

    ConstArray1D(const float* data, int32_t d1) : data_(data), d1_(d1) {}

    virtual ~ConstArray1D();

    // Init the object, the object does not own the data nor copy.
    // It is designed to directly use data from memory mapped resources.
    void init(const int32_t* data, int32_t d1) {
        U_ASSERT(IEEE_754 == 1);
        data_ = reinterpret_cast<const float*>(data);
        d1_ = d1;
    }

    // ReadArray1D methods.
    virtual int32_t d1() const override { return d1_; }
    virtual float get(int32_t i) const override {
        U_ASSERT(i < d1_);
        return data_[i];
    }
    virtual const float* getData() const override { return data_; }
    virtual size_t dataSize() const override { return sizeof(float) * d1(); }

private:
    const float* data_;
    int32_t d1_;
};

ConstArray1D::~ConstArray1D()
{
}

/**
 * A class to index a float array as a 2D Array without owning the pointer or
 * copy the data.
 */
class ConstArray2D : public ReadArray2D {
public:
    ConstArray2D() : data_(nullptr), d1_(0), d2_(0) {}

    ConstArray2D(const float* data, int32_t d1, int32_t d2)
        : data_(data), d1_(d1), d2_(d2) {}

    virtual ~ConstArray2D();

    // Init the object, the object does not own the data nor copy.
    // It is designed to directly use data from memory mapped resources.
    void init(const int32_t* data, int32_t d1, int32_t d2) {
        U_ASSERT(IEEE_754 == 1);
        data_ = reinterpret_cast<const float*>(data);
        d1_ = d1;
        d2_ = d2;
    }

    // ReadArray2D methods.
    inline int32_t d1() const override { return d1_; }
    inline int32_t d2() const override { return d2_; }
    float get(int32_t i, int32_t j) const override {
        U_ASSERT(i < d1_);
        U_ASSERT(j < d2_);
        return data_[i * d2_ + j];
    }
    virtual const float* getData() const override { return data_; }
    virtual size_t dataSize() const override { return sizeof(float) * d1() * d2(); }

    // Expose the ith row as a ConstArray1D
    inline ConstArray1D row(int32_t i) const {
        U_ASSERT(i < d1_);
        return ConstArray1D(data_ + i * d2_, d2_);
    }

private:
    const float* data_;
    int32_t d1_;
    int32_t d2_;
};

ConstArray2D::~ConstArray2D()
{
}

/**
 * A class to allocate data as a writable 1D array.
 * This is the main class implement matrix operation.
 */
class Array1D : public ReadArray1D {
public:
    Array1D() : memory_(nullptr), data_(nullptr), d1_(0) {}
    Array1D(int32_t d1, UErrorCode &status)
        : memory_(uprv_malloc(d1 * sizeof(float))),
          data_((float*)memory_), d1_(d1) {
        if (U_SUCCESS(status)) {
            if (memory_ == nullptr) {
                status = U_MEMORY_ALLOCATION_ERROR;
                return;
            }
            clear();
        }
    }

    virtual ~Array1D();

    // A special constructor which does not own the memory but writeable
    // as a slice of an array.
    Array1D(float* data, int32_t d1)
        : memory_(nullptr), data_(data), d1_(d1) {}

    // ReadArray1D methods.
    virtual int32_t d1() const override { return d1_; }
    virtual float get(int32_t i) const override {
        U_ASSERT(i < d1_);
        return data_[i];
    }
    virtual const float* getData() const override { return data_; }
    virtual size_t dataSize() const override { return sizeof(float) * d1(); }

    // Return the index which point to the max data in the array.
    inline int32_t maxIndex() const {
        int32_t index = 0;
        float max = data_[0];
        for (int32_t i = 1; i < d1_; i++) {
            if (data_[i] > max) {
                max = data_[i];
                index = i;
            }
        }
        return index;
    }

    // Slice part of the array to a new one.
    inline Array1D slice(int32_t from, int32_t size) const {
        U_ASSERT(from >= 0);
        U_ASSERT(from < d1_);
        U_ASSERT(from + size <= d1_);
        return Array1D(data_ + from, size);
    }

    // Add dot product of a 1D array and a 2D array into this one.
    inline Array1D& addDotProduct(const ReadArray1D& a, const ReadArray2D& b) {
        U_ASSERT(a.d1() == b.d1());
        U_ASSERT(b.d2() == d1());
        for (int32_t i = 0; i < d1(); i++) {
            for (int32_t j = 0; j < a.d1(); j++) {
                data_[i] += a.get(j) * b.get(j, i);
            }
        }
        return *this;
    }

    // Hadamard Product the values of another array of the same size into this one.
    inline Array1D& hadamardProduct(const ReadArray1D& a) {
        U_ASSERT(a.d1() == d1());
        for (int32_t i = 0; i < d1(); i++) {
            data_[i] *= a.get(i);
        }
        return *this;
    }

    // Add the Hadamard Product of two arrays of the same size into this one.
    inline Array1D& addHadamardProduct(const ReadArray1D& a, const ReadArray1D& b) {
        U_ASSERT(a.d1() == d1());
        U_ASSERT(b.d1() == d1());
        for (int32_t i = 0; i < d1(); i++) {
            data_[i] += a.get(i) * b.get(i);
        }
        return *this;
    }

    // Add the values of another array of the same size into this one.
    inline Array1D& add(const ReadArray1D& a) {
        U_ASSERT(a.d1() == d1());
        for (int32_t i = 0; i < d1(); i++) {
            data_[i] += a.get(i);
        }
        return *this;
    }

    // Assign the values of another array of the same size into this one.
    inline Array1D& assign(const ReadArray1D& a) {
        U_ASSERT(a.d1() == d1());
        uprv_memcpy((void*)getData(), (void*)a.getData(), d1() * sizeof(float));
        return *this;
    }

    // Apply tanh to all the elements in the array.
    inline Array1D& tanh() {
        return tanh(*this);
    }

    // Apply tanh of a and store into this array.
    inline Array1D& tanh(const Array1D& a) {
        U_ASSERT(a.d1() == d1());
        for (int32_t i = 0; i < d1_; i++) {
            data_[i] = std::tanh(a.get(i));
        }
        return *this;
    }

    // Apply sigmoid to all the elements in the array.
    inline Array1D& sigmoid() {
        for (int32_t i = 0; i < d1_; i++) {
            data_[i] = 1.0f/(1.0f + expf(-data_[i]));
        }
        return *this;
    }

    inline Array1D& clear() {
        uprv_memset(data_, 0, d1_ * sizeof(float));
        return *this;
    }

private:
    void* memory_;
    float* data_;
    int32_t d1_;
};

Array1D::~Array1D()
{
    uprv_free(memory_);
}

class Array2D : public ReadArray2D {
public:
    Array2D() : memory_(nullptr), data_(nullptr), d1_(0), d2_(0) {}
    Array2D(int32_t d1, int32_t d2, UErrorCode &status)
        : memory_(uprv_malloc(d1 * d2 * sizeof(float))),
          data_((float*)memory_), d1_(d1), d2_(d2) {
        if (U_SUCCESS(status)) {
            if (memory_ == nullptr) {
                status = U_MEMORY_ALLOCATION_ERROR;
                return;
            }
            clear();
        }
    }
    virtual ~Array2D();

    // ReadArray2D methods.
    virtual int32_t d1() const override { return d1_; }
    virtual int32_t d2() const override { return d2_; }
    virtual float get(int32_t i, int32_t j) const override {
        U_ASSERT(i < d1_);
        U_ASSERT(j < d2_);
        return data_[i * d2_ + j];
    }

    inline Array1D row(int32_t i) const {
        U_ASSERT(i < d1_);
        return Array1D(data_ + i * d2_, d2_);
    }

    inline Array2D& clear() {
        uprv_memset(data_, 0, d1_ * d2_ * sizeof(float));
        return *this;
    }
    virtual const float* getData() const override { return data_; }
    virtual size_t dataSize() const override { return sizeof(float) * d1() * d2(); }

private:
    void* memory_;
    float* data_;
    int32_t d1_;
    int32_t d2_;
};

Array2D::~Array2D()
{
    uprv_free(memory_);
}

typedef enum {
    BEGIN,
    INSIDE,
    END,
    SINGLE
} LSTMClass;

typedef enum {
    UNKNOWN,
    CODE_POINTS,
    GRAPHEME_CLUSTER,
} EmbeddingType;

struct LSTMData : public UMemory {
    LSTMData(UResourceBundle* rb, UErrorCode &status);
    ~LSTMData();
    UHashtable* fDict;
    EmbeddingType fType;
    const char16_t* fName;
    ConstArray2D fEmbedding;
    ConstArray2D fForwardW;
    ConstArray2D fForwardU;
    ConstArray1D fForwardB;
    ConstArray2D fBackwardW;
    ConstArray2D fBackwardU;
    ConstArray1D fBackwardB;
    ConstArray2D fOutputW;
    ConstArray1D fOutputB;

private:
    UResourceBundle* fBundle;
};

LSTMData::LSTMData(UResourceBundle* rb, UErrorCode &status)
    : fDict(nullptr), fType(UNKNOWN), fName(nullptr),
      fBundle(rb)
{
    if (U_FAILURE(status)) {
        return;
    }
    if (IEEE_754 != 1) {
        status = U_UNSUPPORTED_ERROR;
        return;
    }
    LocalUResourceBundlePointer embeddings_res(
        ures_getByKey(rb, "embeddings", nullptr, &status));
    int32_t embedding_size = ures_getInt(embeddings_res.getAlias(), &status);
    LocalUResourceBundlePointer hunits_res(
        ures_getByKey(rb, "hunits", nullptr, &status));
    if (U_FAILURE(status)) return;
    int32_t hunits = ures_getInt(hunits_res.getAlias(), &status);
    const char16_t* type = ures_getStringByKey(rb, "type", nullptr, &status);
    if (U_FAILURE(status)) return;
    if (u_strCompare(type, -1, u"codepoints", -1, false) == 0) {
        fType = CODE_POINTS;
    } else if (u_strCompare(type, -1, u"graphclust", -1, false) == 0) {
        fType = GRAPHEME_CLUSTER;
    }
    fName = ures_getStringByKey(rb, "model", nullptr, &status);
    LocalUResourceBundlePointer dataRes(ures_getByKey(rb, "data", nullptr, &status));
    if (U_FAILURE(status)) return;
    int32_t data_len = 0;
    const int32_t* data = ures_getIntVector(dataRes.getAlias(), &data_len, &status);
    fDict = uhash_open(uhash_hashUChars, uhash_compareUChars, nullptr, &status);

    StackUResourceBundle stackTempBundle;
    ResourceDataValue value;
    ures_getValueWithFallback(rb, "dict", stackTempBundle.getAlias(), value, status);
    ResourceArray stringArray = value.getArray(status);
    int32_t num_index = stringArray.getSize();
    if (U_FAILURE(status)) { return; }

    // put dict into hash
    int32_t stringLength;
    for (int32_t idx = 0; idx < num_index; idx++) {
        stringArray.getValue(idx, value);
        const char16_t* str = value.getString(stringLength, status);
        uhash_putiAllowZero(fDict, (void*)str, idx, &status);
        if (U_FAILURE(status)) return;
#ifdef LSTM_VECTORIZER_DEBUG
        printf("Assign [");
        while (*str != 0x0000) {
            printf("U+%04x ", *str);
            str++;
        }
        printf("] map to %d\n", idx-1);
#endif
    }
    int32_t mat1_size = (num_index + 1) * embedding_size;
    int32_t mat2_size = embedding_size * 4 * hunits;
    int32_t mat3_size = hunits * 4 * hunits;
    int32_t mat4_size = 4 * hunits;
    int32_t mat5_size = mat2_size;
    int32_t mat6_size = mat3_size;
    int32_t mat7_size = mat4_size;
    int32_t mat8_size = 2 * hunits * 4;
#if U_DEBUG
    int32_t mat9_size = 4;
    U_ASSERT(data_len == mat1_size + mat2_size + mat3_size + mat4_size + mat5_size +
        mat6_size + mat7_size + mat8_size + mat9_size);
#endif

    fEmbedding.init(data, (num_index + 1), embedding_size);
    data += mat1_size;
    fForwardW.init(data, embedding_size, 4 * hunits);
    data += mat2_size;
    fForwardU.init(data, hunits, 4 * hunits);
    data += mat3_size;
    fForwardB.init(data, 4 * hunits);
    data += mat4_size;
    fBackwardW.init(data, embedding_size, 4 * hunits);
    data += mat5_size;
    fBackwardU.init(data, hunits, 4 * hunits);
    data += mat6_size;
    fBackwardB.init(data, 4 * hunits);
    data += mat7_size;
    fOutputW.init(data, 2 * hunits, 4);
    data += mat8_size;
    fOutputB.init(data, 4);
}

LSTMData::~LSTMData() {
    uhash_close(fDict);
    ures_close(fBundle);
}

class Vectorizer : public UMemory {
public:
    Vectorizer(UHashtable* dict) : fDict(dict) {}
    virtual ~Vectorizer();
    virtual void vectorize(UText *text, int32_t startPos, int32_t endPos,
                           UVector32 &offsets, UVector32 &indices,
                           UErrorCode &status) const = 0;
protected:
    int32_t stringToIndex(const char16_t* str) const {
        UBool found = false;
        int32_t ret = uhash_getiAndFound(fDict, (const void*)str, &found);
        if (!found) {
            ret = fDict->count;
        }
#ifdef LSTM_VECTORIZER_DEBUG
        printf("[");
        while (*str != 0x0000) {
            printf("U+%04x ", *str);
            str++;
        }
        printf("] map to %d\n", ret);
#endif
        return ret;
    }

private:
    UHashtable* fDict;
};

Vectorizer::~Vectorizer()
{
}

class CodePointsVectorizer : public Vectorizer {
public:
    CodePointsVectorizer(UHashtable* dict) : Vectorizer(dict) {}
    virtual ~CodePointsVectorizer();
    virtual void vectorize(UText *text, int32_t startPos, int32_t endPos,
                           UVector32 &offsets, UVector32 &indices,
                           UErrorCode &status) const override;
};

CodePointsVectorizer::~CodePointsVectorizer()
{
}

void CodePointsVectorizer::vectorize(
    UText *text, int32_t startPos, int32_t endPos,
    UVector32 &offsets, UVector32 &indices, UErrorCode &status) const
{
    if (offsets.ensureCapacity(endPos - startPos, status) &&
            indices.ensureCapacity(endPos - startPos, status)) {
        if (U_FAILURE(status)) return;
        utext_setNativeIndex(text, startPos);
        int32_t current;
        char16_t str[2] = {0, 0};
        while (U_SUCCESS(status) &&
               (current = (int32_t)utext_getNativeIndex(text)) < endPos) {
            // Since the LSTMBreakEngine is currently only accept chars in BMP,
            // we can ignore the possibility of hitting supplementary code
            // point.
            str[0] = (char16_t) utext_next32(text);
            U_ASSERT(!U_IS_SURROGATE(str[0]));
            offsets.addElement(current, status);
            indices.addElement(stringToIndex(str), status);
        }
    }
}

class GraphemeClusterVectorizer : public Vectorizer {
public:
    GraphemeClusterVectorizer(UHashtable* dict)
        : Vectorizer(dict)
    {
    }
    virtual ~GraphemeClusterVectorizer();
    virtual void vectorize(UText *text, int32_t startPos, int32_t endPos,
                           UVector32 &offsets, UVector32 &indices,
                           UErrorCode &status) const override;
};

GraphemeClusterVectorizer::~GraphemeClusterVectorizer()
{
}

constexpr int32_t MAX_GRAPHEME_CLSTER_LENGTH = 10;

void GraphemeClusterVectorizer::vectorize(
    UText *text, int32_t startPos, int32_t endPos,
    UVector32 &offsets, UVector32 &indices, UErrorCode &status) const
{
    if (U_FAILURE(status)) return;
    if (!offsets.ensureCapacity(endPos - startPos, status) ||
            !indices.ensureCapacity(endPos - startPos, status)) {
        return;
    }
    if (U_FAILURE(status)) return;
    LocalPointer<BreakIterator> graphemeIter(BreakIterator::createCharacterInstance(Locale(), status));
    if (U_FAILURE(status)) return;
    graphemeIter->setText(text, status);
    if (U_FAILURE(status)) return;

    if (startPos != 0) {
        graphemeIter->preceding(startPos);
    }
    int32_t last = startPos;
    int32_t current = startPos;
    char16_t str[MAX_GRAPHEME_CLSTER_LENGTH];
    while ((current = graphemeIter->next()) != BreakIterator::DONE) {
        if (current >= endPos) {
            break;
        }
        if (current > startPos) {
            utext_extract(text, last, current, str, MAX_GRAPHEME_CLSTER_LENGTH, &status);
            if (U_FAILURE(status)) return;
            offsets.addElement(last, status);
            indices.addElement(stringToIndex(str), status);
            if (U_FAILURE(status)) return;
        }
        last = current;
    }
    if (U_FAILURE(status) || last >= endPos) {
        return;
    }
    utext_extract(text, last, endPos, str, MAX_GRAPHEME_CLSTER_LENGTH, &status);
    if (U_SUCCESS(status)) {
        offsets.addElement(last, status);
        indices.addElement(stringToIndex(str), status);
    }
}

// Computing LSTM as stated in
// https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
// ifco is temp array allocate outside which does not need to be
// input/output value but could avoid unnecessary memory alloc/free if passing
// in.
void compute(
    const ReadArray2D& W, const ReadArray2D& U, const ReadArray1D& b,
    const ConstArray2D& E, int32_t row,
    Array1D& h, Array1D& c,
    Array1D& ifco)
{
    int32_t hunits = U.d1();
    ConstArray1D x = E.row(row);
    // ifco = x * W + h * U + b
    ifco.assign(b)
        .addDotProduct(x, W)
        .addDotProduct(h, U);

#if 0
    printf("\nIFCO %d\n", hunits);
    printf("I:\n");
    for (int32_t i = 0; i < hunits; i++) {
        printf("%f ", ifco.get(i));
    }
    printf("\nF:\n");
    for (int32_t i = 0; i < hunits; i++) {
        printf("%f ", ifco.get(i+hunits));
    }
    printf("\nC:\n");
    for (int32_t i = 0; i < hunits; i++) {
        printf("%f ", ifco.get(i+hunits*2));
    }
    printf("\nO:\n");
    for (int32_t i = 0; i < hunits; i++) {
        printf("%f ", ifco.get(i+hunits*3));
    }
#endif

    ifco.slice(0*hunits, hunits).sigmoid();  // i: sigmoid
    ifco.slice(1*hunits, hunits).sigmoid(); // f: sigmoid
    ifco.slice(2*hunits, hunits).tanh(); // c_: tanh
    ifco.slice(3*hunits, hunits).sigmoid(); // o: sigmoid

    c.hadamardProduct(ifco.slice(hunits, hunits))
        .addHadamardProduct(ifco.slice(0, hunits), ifco.slice(2*hunits, hunits));

    h.tanh(c)
        .hadamardProduct(ifco.slice(3*hunits, hunits));

    /*
    printf("\nc:\n");
    for (int32_t i = 0; i < c.d1(); i++) {
        printf("%f ", c.get(i));
    }
    printf("\nh:\n");
    for (int32_t i = 0; i < h.d1(); i++) {
        printf("%f ", h.get(i));
    }
    */
}

void backwardLSTM(const LSTMData* fData, int32_t row, Array1D& h, Array1D& c, Array1D& ifco) {
    compute(fData->fBackwardW, fData->fBackwardU, fData->fBackwardB,
            fData->fEmbedding, row,
            h, c, ifco);
}

void forwardLSTM(const LSTMData* fData, int32_t row, Array1D& h, Array1D& c, Array1D& ifco) {
    compute(fData->fForwardW, fData->fForwardU, fData->fForwardB,
            fData->fEmbedding, row,
            h, c, ifco);
}

void bidirectionalLSTMCPU(const LSTMData* fData, int32_t len, int32_t* input, int32_t* output, UErrorCode& status) {
    double start = uprv_getUTCtime();
    int32_t hunits = fData->fForwardU.d1();
    Array2D h(len, hunits, status);
    Array1D c(hunits, status);
    Array1D ifco(4 * hunits, status);
    Array1D logp(4, status);
    Array1D both(2 * hunits, status);
    Array1D f = both.slice(0, hunits);  // point to first half of data in both.
    Array1D b = both.slice(hunits, hunits);  // point to second half of data in both.

    for (int32_t i = len - 1; i >= 0; i--) {
        backwardLSTM(fData, input[i], b, c, ifco);
        h.row(i).assign(b);
    }

#if 0
    printf("CPU After Backward Scan h:\n");
    for (int i = 0; i < h.d1(); i++) {
      printf("%d: [", i);
      for (int j = 0; j < h.d2(); j++) {
        printf("%f ", h.get(i, j));
      }
      printf("]\n");
    }
#endif

    c.clear();

    // The following iteration merge the forward LSTM and the output layer
    // together.
    for (int32_t i = 0; i < len; i++) {
//        printf("Forward %d = row %d\n", i, input[i]);
        // Forward LSTM
        // Calculate the result into f, which point to the data in the first half
        // of both.
        forwardLSTM(fData, input[i], f, c, ifco);
#if 0
    printf("\ncpu ifco:\n");
    for (int i = 0; i < ifco.d1(); i++) {
        printf("%f ", ifco.get(i));
    }
    printf("\n");
#endif

        // assign the data from hBackward.row(i) to second half of both.
        b.assign(h.row(i));

#if 0
    printf("\ncpu both:\n");
    for (int i = 0; i < both.d1(); i++) {
        printf("%f ", both.get(i));
    }
    printf("\n");
    printf("\ncpu Wo:\n");
    for (int i = 0; i < fData->fOutputW.d1(); i++) {
        for (int j = 0; j < fData->fOutputW.d2(); j++) {
            printf("%f ", fData->fOutputW.get(i, j));
        }
    }
    printf("\ncpu Bo:\n");
    for (int i = 0; i < fData->fOutputB.d1(); i++) {
        printf("%f ", fData->fOutputB.get(i));
    }
    printf("\n");
#endif
        // logp = fData->fOutputB + both * fData->fOutputW
        logp.assign(fData->fOutputB).addDotProduct(both, fData->fOutputW);

        // current = argmax(logp)
        output[i] = logp.maxIndex();
    }
    double end = uprv_getUTCtime();
    printf("CPU %f\n", end-start);
}

cl_mem create2DData(cl_context cx, cl_command_queue q, const ConstArray2D& d, int& err) {
    cl_mem mem = clCreateBuffer(cx, CL_MEM_READ_ONLY, d.dataSize(), NULL, NULL);
    err = clEnqueueWriteBuffer(q, mem, CL_TRUE, 0, d.dataSize(), d.getData(), 0, NULL, NULL);
    U_ASSERT(err == 0);
    return mem;
}
cl_mem create1DData(cl_context cx, cl_command_queue q, const ConstArray1D& d, int& err) {
    cl_mem mem = clCreateBuffer(cx, CL_MEM_READ_ONLY, d.dataSize(), NULL, NULL);
    err = clEnqueueWriteBuffer(q, mem, CL_TRUE, 0, d.dataSize(), d.getData(), 0, NULL, NULL);
    return mem;
}
cl_mem createBuffer(cl_context cx, int32_t len) {
    return clCreateBuffer(cx, CL_MEM_READ_WRITE, sizeof(float) * len, NULL, NULL);
}
cl_mem createIntBuffer(cl_context cx, int32_t len) {
    return clCreateBuffer(cx, CL_MEM_READ_WRITE, sizeof(int) * len, NULL, NULL);
}

size_t round_work_group_size(size_t items) {
  const size_t kMinWorkGroupSize = 512;
  size_t global = kMinWorkGroupSize;
  if (items > kMinWorkGroupSize) {
      // round the multiple of kMinWorkGroupSize
      global = items - (items % kMinWorkGroupSize) + ((items % kMinWorkGroupSize == 0) ? 0 : kMinWorkGroupSize);
  }
  return global;
}

void add(cl_device_id id, cl_command_queue q, cl_mem out, cl_mem a, cl_mem b, int len, int offset) {
    size_t local;
    size_t global = round_work_group_size(len);
    cl_kernel k = gOpenCL->getAdd();
    int err = clSetKernelArg(k, 0, sizeof(cl_mem), &out);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 1, sizeof(cl_mem), &a);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 2, sizeof(cl_mem), &b);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 3, sizeof(int), &len);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 4, sizeof(int), &offset);
    U_ASSERT(err == 0);
    err = clGetKernelWorkGroupInfo(k, id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    U_ASSERT(err == 0);
    err = clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL);
    U_ASSERT(err == 0);
}
void maxIndex(cl_device_id id, cl_command_queue q, cl_mem out, cl_mem in, int len, int group) {
    size_t local;
    size_t global = round_work_group_size(len);
    cl_kernel k = gOpenCL->getMaxIndex();
    int err = clSetKernelArg(k, 0, sizeof(cl_mem), &out);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 1, sizeof(cl_mem), &in);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 2, sizeof(int), &len);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 3, sizeof(int), &group);
    U_ASSERT(err == 0);
    err = clGetKernelWorkGroupInfo(k, id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    U_ASSERT(err == 0);
    err = clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL);
    U_ASSERT(err == 0);
}
void add3(cl_device_id id, cl_command_queue q, cl_mem out, cl_mem a, cl_mem b, cl_mem c, int len) {
    size_t local;
    size_t global = round_work_group_size(len);
    cl_kernel k = gOpenCL->getAdd3();
    int err = clSetKernelArg(k, 0, sizeof(cl_mem), &out);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 1, sizeof(cl_mem), &a);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 2, sizeof(cl_mem), &b);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 3, sizeof(cl_mem), &c);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 4, sizeof(int), &len);
    U_ASSERT(err == 0);
    err = clGetKernelWorkGroupInfo(k, id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    U_ASSERT(err == 0);
    err = clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL);
    U_ASSERT(err == 0);
}
void mul1d2d(cl_device_id id, cl_command_queue q, cl_mem out, cl_mem a, cl_mem b,
             int n, int m, int aOffset) {
    size_t local;
    size_t global = round_work_group_size(n*m);
    cl_kernel k = gOpenCL->getMul1d2d();
    int err = clSetKernelArg(k, 0, sizeof(cl_mem), &out);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 1, sizeof(cl_mem), &a);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 2, sizeof(cl_mem), &b);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 3, sizeof(int), &n);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 4, sizeof(int), &m);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 5, sizeof(int), &aOffset);
    U_ASSERT(err == 0);
    err = clGetKernelWorkGroupInfo(k, id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    U_ASSERT(err == 0);
    err = clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL);
    U_ASSERT(err == 0);
}
void print(cl_device_id id, cl_command_queue q, cl_mem a, int len, int offset) {
    size_t local;
    size_t global = round_work_group_size(len);
    cl_kernel k = gOpenCL->getPrint();
    int err = clSetKernelArg(k, 0, sizeof(cl_mem), &a);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 1, sizeof(int), &len);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 2, sizeof(int), &offset);
    U_ASSERT(err == 0);
    err = clGetKernelWorkGroupInfo(k, id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    U_ASSERT(err == 0);
    err = clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL);
    U_ASSERT(err == 0);
}

void ident(cl_device_id id, cl_command_queue q, cl_mem a, int len, int offset) {
    size_t local;
    size_t global = round_work_group_size(len);
    cl_kernel k = gOpenCL->getIdent();
    int err = clSetKernelArg(k, 0, sizeof(cl_mem), &a);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 1, sizeof(int), &len);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 2, sizeof(int), &offset);
    U_ASSERT(err == 0);
    err = clGetKernelWorkGroupInfo(k, id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    U_ASSERT(err == 0);
    err = clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL);
    U_ASSERT(err == 0);
}
void clear(cl_device_id id, cl_command_queue q, cl_mem a, int len, int offset) {
    size_t local;
    size_t global = round_work_group_size(len);
    cl_kernel k = gOpenCL->getClear();
    int err = clSetKernelArg(k, 0, sizeof(cl_mem), &a);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 1, sizeof(int), &len);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 2, sizeof(int), &offset);
    U_ASSERT(err == 0);
    err = clGetKernelWorkGroupInfo(k, id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    U_ASSERT(err == 0);
    err = clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL);
    U_ASSERT(err == 0);
}

void offsetAdd(cl_device_id id, cl_command_queue q, cl_mem data, int length, int offsetA, int offsetB) {
    size_t local;
    cl_kernel k = gOpenCL->getOffsetAdd();
    size_t global = round_work_group_size(length);
    int err = clSetKernelArg(k, 0, sizeof(cl_mem), &data);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 1, sizeof(int), &length);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 2, sizeof(int), &offsetA);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 3, sizeof(int), &offsetB);
    U_ASSERT(err == 0);
    err = clGetKernelWorkGroupInfo(k, id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    U_ASSERT(err == 0);
    err = clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL);
    U_ASSERT(err == 0);
}

void mul(cl_device_id id, cl_command_queue q, cl_mem c, cl_mem a, cl_mem b,
         int d1, int d2, int aOffset) {
    mul1d2d(id, q, c, a, b, d1, d2, aOffset);
    int n = d1;
    while (n != 1) {                 // Run ceil(log(n)) times
        int aStart = 0;
        int bStart = ((n+1)/2) * d2; // ceil(n/2) * d2
        offsetAdd(id, q, c, (n/2) * d2 , aStart, bStart);
        n = (n+1)/2;  // n = ceil(n/2)
    }
}

void tanhOrSigmoid(cl_device_id id, cl_context cx, cl_command_queue q, cl_mem ifco, int length, int hunits) {
    size_t local;
    size_t global = round_work_group_size(length);
    cl_kernel k = gOpenCL->getTanhOrSigmoid();
    int err = clSetKernelArg(k, 0, sizeof(cl_mem), &ifco);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 1, sizeof(int), &length);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 2, sizeof(int), &hunits);
    U_ASSERT(err == 0);
    err = clGetKernelWorkGroupInfo(k, id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    U_ASSERT(err == 0);
    err = clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL);
    U_ASSERT(err == 0);
}

void updateCandH(cl_device_id id, cl_context cx, cl_command_queue q, cl_mem ifco, cl_mem c, cl_mem h, int hunits) {
    size_t local;
    size_t global = round_work_group_size(hunits);
    const int iOffset = 0;
    const int fOffset = hunits;
    const int cOffset = 2 * hunits;
    const int oOffset = 3 * hunits;
    cl_kernel k = gOpenCL->getUpdateCH();
    int err = clSetKernelArg(k, 0, sizeof(cl_mem), &ifco);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 1, sizeof(cl_mem), &c);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 2, sizeof(cl_mem), &h);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 3, sizeof(int), &hunits);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 4, sizeof(int), &iOffset);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 5, sizeof(int), &fOffset);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 6, sizeof(int), &cOffset);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 7, sizeof(int), &oOffset);
    U_ASSERT(err == 0);
    err = clGetKernelWorkGroupInfo(k, id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    U_ASSERT(err == 0);
    err = clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL);
    U_ASSERT(err == 0);
}

void Output(cl_device_id id, cl_context cx, cl_command_queue q,
            cl_mem out, cl_mem both, cl_mem W, cl_mem B,
            int hunits, int offset) {
    // The bothW should be in (1, 4) dimension, but the intermediate step
    // need dimension of (2*hunits, 4) so we allocate that but only take
    // the first (1, 4) from the result.
    cl_mem bothW = createBuffer(cx, 8*hunits);
    mul(id, q, bothW, both, W, 2*hunits, 4, 0);
    // out = B + both x W
    add(id, q, out, B, bothW, 4, offset);
    clReleaseMemObject(bothW);
}

void LSTMIter(cl_device_id id, cl_context cx, cl_command_queue q,
              cl_mem W, cl_mem U, cl_mem B, cl_mem E, cl_mem h, cl_mem ifco, cl_mem c,
              int hunits, int embeddings, int row) {
    int h4 = 4 * hunits;
    // The xW should be in (1, 4hunits) dimension, but the intermediate step
    // need dimension of (embeddings, 4hunits) so we allocate that but only take
    // the first (1, 4hunts) from the result.
    cl_mem xW = createBuffer(cx, embeddings * h4);
    mul(id, q, xW, E, W, embeddings, h4, row*embeddings);

    // The xW should be in (1, 4hunits) dimension, but the intermediate step
    // need dimension of (h, 4hunits) so we allocate that but only take
    // the first (1, 4hunts) from the result.
    cl_mem hU = createBuffer(cx, hunits * h4);
    mul(id, q, hU, h, U, hunits, h4, 0);
    add3(id, q, ifco, B, xW, hU, h4);
    clReleaseMemObject(hU);
    clReleaseMemObject(xW);
    tanhOrSigmoid(id, cx, q, ifco, 4*hunits, hunits);
    updateCandH(id, cx, q, ifco, c, h, hunits);
}

void copy(cl_device_id id, cl_command_queue q, cl_mem dest, cl_mem src, int length, int destOffset, int srcOffset) {
    size_t local;
    size_t global = 4096;
    cl_kernel k = gOpenCL->getCopy();
    int err = clSetKernelArg(k, 0, sizeof(cl_mem), &dest);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 1, sizeof(cl_mem), &src);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 2, sizeof(int), &length);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 3, sizeof(int), &destOffset);
    U_ASSERT(err == 0);
    err = clSetKernelArg(k, 4, sizeof(int), &srcOffset);
    U_ASSERT(err == 0);
    err = clGetKernelWorkGroupInfo(k, id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    U_ASSERT(err == 0);
    err = clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL);
    U_ASSERT(err == 0);
}

void bidirectionalLSTMOpenCL(const LSTMData* fData, int32_t len, int32_t* input, int32_t* output, UErrorCode& status) {
    umtx_initOnce(gOpenCLInitOnce, &initOpenCL, status);
    double start = uprv_getUTCtime();

    int err;
    cl_device_id id = gOpenCL->getDeviceID();
    cl_context cx = gOpenCL->getContext();
    cl_command_queue q = clCreateCommandQueue(cx, id, 0, &err);

    int hunits = fData->fForwardU.d1();

    cl_mem E = create2DData(cx, q, fData->fEmbedding, err);
    U_ASSERT(err == 0);
    cl_mem W = create2DData(cx, q, fData->fBackwardW, err);
    U_ASSERT(err == 0);
    cl_mem U = create2DData(cx, q, fData->fBackwardU, err);
    U_ASSERT(err == 0);
    cl_mem B = create1DData(cx, q, fData->fBackwardB, err);
    U_ASSERT(err == 0);

    // we creaet both with 2 * hunits but we always use only the first hunits
    // in LSTM and only use the 2nd hunits when calculate the logp
    // in the forward step.
    cl_mem both = createBuffer(cx, 2 * hunits);
    cl_mem h = createBuffer(cx, len * hunits);
    cl_mem c = createBuffer(cx, hunits);
    cl_mem ifco = createBuffer(cx, 4 * hunits);


    const int embeddings = fData->fEmbedding.d2();
    for (int32_t index = len - 1; index >= 0; index--) {
        const int row = input[index];
        LSTMIter(id, cx, q, W, U, B, E, both, ifco, c,
                 hunits, embeddings, row);
        copy(id, q, h, both, hunits, index*hunits, 0);
    }

    clReleaseMemObject(W);
    clReleaseMemObject(U);
    clReleaseMemObject(B);

    W = create2DData(cx, q, fData->fForwardW, err);
    U_ASSERT(err == 0);
    U = create2DData(cx, q, fData->fForwardU, err);
    U_ASSERT(err == 0);
    B = create1DData(cx, q, fData->fForwardB, err);
    U_ASSERT(err == 0);
    cl_mem oW = create2DData(cx, q, fData->fOutputW, err);
    U_ASSERT(err == 0);
    cl_mem oB = create1DData(cx, q, fData->fOutputB, err);
    U_ASSERT(err == 0);

    cl_mem logp = createBuffer(cx, 4*len);

#if 0
    printf("After Backward Scan: GPU h\n");
    print(id, q, h, len*hunits, 0);
#endif

    clear(id, q, c, hunits, 0);
    clear(id, q, both, hunits, 0); // clear the first half
    for (int32_t index = 0; index < len; index++) {
        const int row = input[index];
        LSTMIter(id, cx, q,
                 W, U, B, E, both, ifco, c,
                 hunits, embeddings, row);
        // copy from h[index] to the second half of b.
        copy(id, q, both, h, hunits, hunits, index*hunits);
        Output(id, cx, q, logp, both, oW, oB, hunits, index * 4);
    }
    clReleaseMemObject(h);
    clReleaseMemObject(c);
    clReleaseMemObject(both);
    clReleaseMemObject(ifco);
    clReleaseMemObject(E);
    clReleaseMemObject(W);
    clReleaseMemObject(U);
    clReleaseMemObject(B);
    clReleaseMemObject(oW);
    clReleaseMemObject(oB);

    cl_mem index = createIntBuffer(cx, len);
    maxIndex(id, q, index, logp, len, 4);

    // To Debug
    err = clFinish(q);
    U_ASSERT(err == 0);

    err = clEnqueueReadBuffer(q, index, CL_TRUE, 0, len * sizeof(int), output, 0, NULL, NULL);

    U_ASSERT(err == 0);

    clReleaseMemObject(logp);
    clReleaseMemObject(index);

    clReleaseCommandQueue(q);
    double end = uprv_getUTCtime();
    printf("OpenCL %f\n", end-start);
}

// Minimum word size
static const int32_t MIN_WORD = 2;

// Minimum number of characters for two words
static const int32_t MIN_WORD_SPAN = MIN_WORD * 2;

int32_t
LSTMBreakEngine::divideUpDictionaryRange( UText *text,
                                                int32_t startPos,
                                                int32_t endPos,
                                                UVector32 &foundBreaks,
                                                UBool /* isPhraseBreaking */,
                                                UErrorCode& status) const {
    if (U_FAILURE(status)) return 0;
    int32_t beginFoundBreakSize = foundBreaks.size();
    utext_setNativeIndex(text, startPos);
    utext_moveIndex32(text, MIN_WORD_SPAN);
    if (utext_getNativeIndex(text) >= endPos) {
        return 0;       // Not enough characters for two words
    }
    utext_setNativeIndex(text, startPos);

    UVector32 offsets(status);
    UVector32 indices(status);
    if (U_FAILURE(status)) return 0;
    fVectorizer->vectorize(text, startPos, endPos, offsets, indices, status);
    if (U_FAILURE(status)) return 0;
    int32_t* offsetsBuf = offsets.getBuffer();

    int32_t len = indices.size();
    // ----- Begin of all the Array memory allocation needed for this function
    // Allocate temp array used inside compute()

    // TODO: limit size of hBackward. If len is too big, we could
    // run out of memory.
    // Backward LSTM

    // ----- End of all the Array memory allocation needed for this function
    if (U_FAILURE(status)) return 0;

    UVector32 m1(len, status);
    m1.setSize(len);
    bidirectionalLSTMCPU(fData, indices.size(), indices.getBuffer(), m1.getBuffer(), status);
    if (U_FAILURE(status)) return 0;

    UVector32 m2(len, status);
    m2.setSize(len);
    bidirectionalLSTMOpenCL(fData, indices.size(), indices.getBuffer(), m2.getBuffer(), status);

    printf("maxIndex and maxIndex2 is %s\n", m1 == m2 ? "SAME" : "Different");
    if (U_FAILURE(status)) return 0;

#if 0
    for (int32_t i = 0; i < len; i++) {
      if (maxIndex.elementAti(i) != maxIndex2.elementAti(i)) {
        printf("Diff %d = %d vs %d\n", i, maxIndex.elementAti(i), maxIndex2.elementAti(i));
      }
    }
#endif

    for (int32_t i = 0; i < len; i++) {
        LSTMClass current = (LSTMClass)m1.elementAti(i);
        // BIES logic.
        if (current == BEGIN || current == SINGLE) {
            if (i != 0) {
                foundBreaks.addElement(offsetsBuf[i], status);
                if (U_FAILURE(status)) return 0;
            }
        }
    }

    int32_t ret = foundBreaks.size() - beginFoundBreakSize;
    return ret;
}

Vectorizer* createVectorizer(const LSTMData* data, UErrorCode &status) {
    if (U_FAILURE(status)) {
        return nullptr;
    }
    switch (data->fType) {
        case CODE_POINTS:
            return new CodePointsVectorizer(data->fDict);
            break;
        case GRAPHEME_CLUSTER:
            return new GraphemeClusterVectorizer(data->fDict);
            break;
        default:
            break;
    }
    UPRV_UNREACHABLE_EXIT;
}

LSTMBreakEngine::LSTMBreakEngine(const LSTMData* data, const UnicodeSet& set, UErrorCode &status)
    : DictionaryBreakEngine(), fData(data), fVectorizer(createVectorizer(fData, status))
{
    if (U_FAILURE(status)) {
      fData = nullptr;  // If failure, we should not delete fData in destructor because the caller will do so.
      return;
    }
    setCharacters(set);
}

LSTMBreakEngine::~LSTMBreakEngine() {
    delete fData;
    delete fVectorizer;
}

const char16_t* LSTMBreakEngine::name() const {
    return fData->fName;
}

UnicodeString defaultLSTM(UScriptCode script, UErrorCode& status) {
    // open root from brkitr tree.
    UResourceBundle *b = ures_open(U_ICUDATA_BRKITR, "", &status);
    b = ures_getByKeyWithFallback(b, "lstm", b, &status);
    UnicodeString result = ures_getUnicodeStringByKey(b, uscript_getShortName(script), &status);
    ures_close(b);
    return result;
}

U_CAPI const LSTMData* U_EXPORT2 CreateLSTMDataForScript(UScriptCode script, UErrorCode& status)
{
    if (script != USCRIPT_KHMER && script != USCRIPT_LAO && script != USCRIPT_MYANMAR && script != USCRIPT_THAI) {
        return nullptr;
    }
    UnicodeString name = defaultLSTM(script, status);
    if (U_FAILURE(status)) return nullptr;
    CharString namebuf;
    namebuf.appendInvariantChars(name, status).truncate(namebuf.lastIndexOf('.'));

    LocalUResourceBundlePointer rb(
        ures_openDirect(U_ICUDATA_BRKITR, namebuf.data(), &status));
    if (U_FAILURE(status)) return nullptr;

    return CreateLSTMData(rb.orphan(), status);
}

U_CAPI const LSTMData* U_EXPORT2 CreateLSTMData(UResourceBundle* rb, UErrorCode& status)
{
    return new LSTMData(rb, status);
}

U_CAPI const LanguageBreakEngine* U_EXPORT2
CreateLSTMBreakEngine(UScriptCode script, const LSTMData* data, UErrorCode& status)
{
    UnicodeString unicodeSetString;
    switch(script) {
        case USCRIPT_THAI:
            unicodeSetString = UnicodeString(u"[[:Thai:]&[:LineBreak=SA:]]");
            break;
        case USCRIPT_MYANMAR:
            unicodeSetString = UnicodeString(u"[[:Mymr:]&[:LineBreak=SA:]]");
            break;
        default:
            delete data;
            return nullptr;
    }
    UnicodeSet unicodeSet;
    unicodeSet.applyPattern(unicodeSetString, status);
    const LanguageBreakEngine* engine = new LSTMBreakEngine(data, unicodeSet, status);
    if (U_FAILURE(status) || engine == nullptr) {
        if (engine != nullptr) {
            delete engine;
        } else {
            status = U_MEMORY_ALLOCATION_ERROR;
        }
        return nullptr;
    }
    return engine;
}

U_CAPI void U_EXPORT2 DeleteLSTMData(const LSTMData* data)
{
    delete data;
}

U_CAPI const char16_t* U_EXPORT2 LSTMDataName(const LSTMData* data)
{
    return data->fName;
}

U_NAMESPACE_END

#endif /* #if !UCONFIG_NO_BREAK_ITERATION */
