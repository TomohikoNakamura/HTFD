/*!
 * @file libfacwt.cpp
 * @brief C++ file for Python wrapper of CWT
 * @date 1 Dec. 2015.
 * @author Tomohiko Nakamura
 */

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/range/value_type.hpp>
#include <common.hpp>
#include <cwt.hpp>
#include <chrono>

namespace py = boost::python;
namespace np = boost::python::numpy;

template <typename T>
np::ndarray convert_vector_to_ndarray(std::vector<T> &vec)
{
  const unsigned int N = vec.size();
  py::tuple shape = py::make_tuple(N);
  np::ndarray ret = np::zeros(shape, np::dtype::get_builtin<T>());
  for (int i = 0; i < N; i++)
  {
    ret[i] = vec[i];
  }
  return ret;
}

template <typename T>
np::ndarray convert_matrix_to_ndarray(std::vector<std::vector<T>> &vec)
{
  const unsigned int N = vec.size(), M = vec[0].size();
  py::tuple shape = py::make_tuple(N, M);
  np::ndarray ret = np::zeros(shape, np::dtype::get_builtin<T>());
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < M; j++)
    {
      ret[i][j] = vec[i][j];
    }
  }
  return ret;
}

template <typename T>
std::vector<T> convert_ndarray_to_vector(np::ndarray &nd)
{
  const unsigned int N = nd.shape(0);
  std::vector<T> vec(N);
  T *data = reinterpret_cast<T *>(nd.get_data());
  for (int i = 0; i < N; i++)
  {
    vec[i] = data[i];
  }
  return vec;
}

template <typename T>
std::vector<std::vector<T>> convert_ndarray_to_matrix(np::ndarray &nd)
{
  const unsigned int N = nd.shape(0);
  const unsigned int M = nd.shape(1);
  std::vector<std::vector<T>> vec(N);
  T *data = reinterpret_cast<T *>(nd.get_data());
  for (int i = 0; i < N; i++)
  {
    vec[i].resize(M);
    for (int j = 0; j < M; j++)
    {
      vec[i][j] = data[i * M + j];
    }
  }
  return vec;
}

/*! @brief initialization of cwt instance
 *
 * @tparam D_TYPE data type
 *
 * @param facwt cwt instance
 * @param _sigLen signal length
 * @param lowFreq lowest frequency
 * @param highFreq highest frequency
 * @param fs sampling rate
 * @param resol resolution of an octave
 * @param width Ratio of standard deviation for approximating frequency responses of the filterbank. The frequency responses are assumed to be zero outside the range [-width\sigma,width\sigma] around each center frequency.
 * @param sd Standard deviation of Gaussian shape in the log-frequency domain.
 * @param alpha If alpha=1 (2), magnitudes (power, resp.) of frequency responses have Gaussian shape.
 * @param multirate If true, setup for multirate spectrogram; otherwise setup for spectrogram with common sampling rate between all channels.
 */
template <typename D_TYPE>
void init_cwt(cwt::CWT<D_TYPE> &facwt,
              const int _sigLen,
              const D_TYPE lowFreq,
              const D_TYPE highFreq,
              const long fs,
              const int resol,
              const D_TYPE width,
              const D_TYPE sd,
              const D_TYPE alpha,
              const bool multirate,
              const int minWidth,
              const std::string waveletType)
{
  facwt.init(_sigLen, lowFreq, highFreq, fs, resol);
  if(waveletType == std::string("log_normal")){
    std::tie(facwt.startFreqIndexes, facwt.freqResps, facwt.sumOfFilterOutputs) =
        cwt::set_lognormal_wavelet<D_TYPE>(facwt.centerAngFreqs, facwt.T, width, sd, alpha, multirate);
  }else if(waveletType == std::string("gabor")){
    std::tie(facwt.startFreqIndexes, facwt.freqResps, facwt.sumOfFilterOutputs) =
        cwt::set_gabor_wavelet<D_TYPE>(facwt.centerAngFreqs, facwt.T, width, sd, alpha, multirate);
  }else{
    throw std::runtime_error("Unknown wavelet type");
  }
  for(auto &v: facwt.freqResps){
    if(v.size()<minWidth){
      while(v.size()!=minWidth){
        v.push_back(0.0);
      }
    }
  }
  facwt.init_ffts();
}

template <typename D_TYPE>
void cwt_forward(py::list &pySpectrogram, const np::ndarray &waveform, cwt::CWT<D_TYPE> &facwt, const int verbose)
{
  // check
  if (waveform.get_nd() != 1)
  {
    throw std::runtime_error("Waveform must be monaural");
  }
  if (waveform.get_dtype() != np::dtype::get_builtin<D_TYPE>())
  {
    throw std::runtime_error("Waveform must be float32");
  }
  // prepare
  const int T = waveform.shape(0);
  std::vector<D_TYPE> data(T);
  std::vector<std::complex<D_TYPE>> spectrum;
  std::vector<std::vector<std::complex<D_TYPE>>> spectrogram;
  // copy data
  std::chrono::system_clock::time_point start, end;
  if (verbose > 0)
  {
    std::cerr << "Copying data: ";
    start = std::chrono::system_clock::now();
  }
  D_TYPE *waveformPtr = reinterpret_cast<D_TYPE *>(waveform.get_data());
  #ifdef _OPNEMP
  #pragma omp parallel for
  #endif
  for (int t = 0; t < T; t++)
  {
    data[t] = waveformPtr[t];
  }
  if (verbose > 0)
  {
    end = std::chrono::system_clock::now();
    std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 100.0 << " s elapsed." << std::endl;
  }
  // forward
  if (verbose > 0)
  {
    std::cerr << "Performing foward: ";
    start = std::chrono::system_clock::now();
  }
  facwt.FFT_signal(spectrum, data, true);
  facwt.forward(spectrogram, spectrum, true, false);
  if (verbose > 0)
  {
    end = std::chrono::system_clock::now();
    std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 100.0 << " s elapsed." << std::endl;
  }
  //
  if (verbose > 0)
  {
    std::cerr << "Copying results: ";
    start = std::chrono::system_clock::now();
  }
  for (int l = 0; l < spectrogram.size(); l++)
  {
    np::ndarray filtered = np::zeros(py::make_tuple(spectrogram[l].size()), np::dtype::get_builtin<std::complex<D_TYPE>>());
    std::complex<D_TYPE> *filteredPtr = reinterpret_cast<std::complex<D_TYPE> *>(filtered.get_data());
#ifdef _OPNEMP
#pragma omp parallel for
#endif
    for (int t = 0; t < spectrogram[l].size(); t++)
    {
      filteredPtr[t] = spectrogram[l][t];
    }
    pySpectrogram.append(filtered);
  }
  if (verbose > 0)
  {
    end = std::chrono::system_clock::now();
    std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 100.0 << " s elapsed." << std::endl;
  }
}

template <typename D_TYPE>
np::ndarray cwt_backward(const py::list &pySpectrogram, cwt::CWT<D_TYPE> &facwt, const int verbose)
{
  // prepare
  const int L = py::len(pySpectrogram);
  std::vector<std::vector<std::complex<D_TYPE>>> spectrogram(L);
  std::vector<D_TYPE> data;
  std::vector<std::complex<D_TYPE>> spectrum;
  // copy spectrogram
  std::chrono::system_clock::time_point start, end;
  if (verbose > 0)
  {
    std::cerr << "Copying data: ";
    start = std::chrono::system_clock::now();
  }
  for (int l = 0; l < L; l++)
  {
    const np::ndarray pySpectrogramSlice = py::extract<np::ndarray>(pySpectrogram[l]);
    //
    if(pySpectrogramSlice.get_dtype() != np::dtype::get_builtin<std::complex<D_TYPE>>()){
      throw std::runtime_error("Spectrogram slice must be complex float");
    }
    //
    std::complex<D_TYPE> *pySpectrogramSlicePtr = reinterpret_cast<std::complex<D_TYPE> *>(pySpectrogramSlice.get_data());
    const int M_l = pySpectrogramSlice.shape(0);
    spectrogram[l].resize(M_l);
    for (int m_l = 0; m_l < M_l; m_l++)
    {
      spectrogram[l][m_l] = pySpectrogramSlicePtr[m_l];
    }
  }
  if (verbose > 0)
  {
    end = std::chrono::system_clock::now();
    std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 100.0 << " s elapsed." << std::endl;
  }
  // backward
  if (verbose > 0)
  {
    std::cerr << "Performing backward: ";
    start = std::chrono::system_clock::now();
  }
  // for (verbose > 0)
  // {
  //   std::cerr << "Performing backward: ";
  //   start = std::chrono::system_clock::now();
  // }
  facwt.backward(spectrum, spectrogram, true, false);
  facwt.IFFT_spectrum(data, spectrum, true);
  //
  if (verbose > 0)
  {
    end = std::chrono::system_clock::now();
    std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 100.0 << " s elapsed." << std::endl;
  }
  // copy data
  if (verbose > 0)
  {
    std::cerr << "Copying results: ";
    start = std::chrono::system_clock::now();
  }
  np::ndarray waveform = np::zeros(py::make_tuple(data.size()), np::dtype::get_builtin<D_TYPE>());
  D_TYPE *waveformPtr = reinterpret_cast<D_TYPE *>(waveform.get_data());
  for (int t = 0; t < data.size(); t++)
  {
    waveformPtr[t] = data[t];
  }
  if (verbose > 0)
  {
    end = std::chrono::system_clock::now();
    std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 100.0 << " s elapsed." << std::endl;
  }
  return waveform;
}

BOOST_PYTHON_MODULE(libfacwt)
{
  // Py_Initialize();
  np::initialize();
  /// vector indexing
  py::class_<std::vector<int>>("IVec")
      .def(py::vector_indexing_suite<std::vector<int>>());
  py::class_<std::vector<float>>("FVec")
      .def(py::vector_indexing_suite<std::vector<float>>());
  py::class_<std::vector<std::vector<float>>>("FVec2d")
      .def(py::vector_indexing_suite<std::vector<std::vector<float>>>());
  py::class_<std::vector<std::complex<float>>>("CFVec")
      .def(py::vector_indexing_suite<std::vector<std::complex<float>>>());
  py::class_<std::vector<std::vector<std::complex<float>>>>("CFVec2d")
      .def(py::vector_indexing_suite<std::vector<std::vector<std::complex<float>>>>());
  /// function definition
  py::class_<cwt::CWT<float>>("cppCWT")
      .def("init", &cwt::CWT<float>::init)
      .def("init_ffts", &cwt::CWT<float>::init_ffts)
      .def("FFT_signal", &cwt::CWT<float>::FFT_signal)
      .def("IFFT_spectrum", &cwt::CWT<float>::IFFT_spectrum)
      .def("forward", &cwt::CWT<float>::forward)
      .def("backward", &cwt::CWT<float>::backward)
      .def("reconstruct_phase", &cwt::CWT<float>::reconstruct_phase)
      .def("save", &cwt::CWT<float>::save)
      .def("load", &cwt::CWT<float>::load)
      .def_readwrite("sigLen", &cwt::CWT<float>::sigLen)
      .def_readwrite("T", &cwt::CWT<float>::T)
      .def_readwrite("start_freq_indexes", &cwt::CWT<float>::startFreqIndexes)
      .def_readwrite("freq_resps", &cwt::CWT<float>::freqResps)
      .def_readwrite("sum_filout", &cwt::CWT<float>::sumOfFilterOutputs)
      .def_readwrite("center_angfreqs", &cwt::CWT<float>::centerAngFreqs);
  py::def("set_lognormal_wavelet", &cwt::set_lognormal_wavelet<float>);
  py::def("set_gabor_wavelet", &cwt::set_gabor_wavelet<float>);
  py::def("init_cwt", &init_cwt<float>);
  py::def("cwt_forward", &cwt_forward<float>);
  py::def("cwt_backward", &cwt_backward<float>);
  // utility functions
  py::def("read_spectrogram", &utils::read_spectrogram<float>);
  py::def("write_spectrogram", &utils::write_spectrogram<float>);
  py::def("vecf2nd", &convert_vector_to_ndarray<float>);
  py::def("matf2nd", &convert_matrix_to_ndarray<float>);
  py::def("nd2vecf", &convert_ndarray_to_vector<float>);
  py::def("nd2matf", &convert_ndarray_to_matrix<float>);
  py::def("veci2nd", &convert_vector_to_ndarray<int>);
  py::def("mati2nd", &convert_matrix_to_ndarray<int>);
  py::def("nd2veci", &convert_ndarray_to_vector<int>);
  py::def("nd2mati", &convert_ndarray_to_matrix<int>);
}
