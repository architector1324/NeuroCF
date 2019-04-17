#pragma once

#include "MatrixCF.hpp"

namespace ncf{
    using namespace mcf;
    using namespace ecl;

    template<typename T>
    class Layer{
    private:
        std::map<std::size_t, Mat<T>> core;
        std::size_t neurons = 0;

        std::function<T(const T&)> activation = nullptr;
        std::function<T(const T&)> derivative = nullptr;

        std::string computer_activation = "";
        std::string computer_derivative = "";

        std::function<void(Mat<T>&)> coregen = nullptr;

    public:
        Layer() = delete;
        explicit Layer(std::size_t);

        void send(Computer&);
        void receive(Computer&);
        void grab(Computer&);
        void release(Computer&);

        template<typename U>
        friend Computer& operator<<(Computer&, Layer<U>&);
        template<typename U>
        friend Computer& operator>>(Computer&, Layer<U>&);

        bool checkCore(std::size_t) const;
        void createCore(std::size_t);
        void releaseCore(std::size_t);

        void setActivation(const std::function<T(const T&)>&);
        void setDerivative(const std::function<T(const T&)>&);

        void setActivation(const std::string&);
        void setDerivative(const std::string&);

        void setCoreGen(const std::function<void(Mat<T>&)>&);

        std::size_t getNeurons() const;
        Mat<T>& getCore(std::size_t);
        const Mat<T>& getConstCore(std::size_t) const;
        const std::function<T(const T&)>& getActivation() const;
        const std::function<T(const T&)>& getDerivative() const;
        const std::string& getComputerActivation() const;
        const std::string& getComputerDerivative() const;
        const std::function<void(Mat<T>&)>& getCoreGen() const;

        // Low-level methods
        void query(const Mat<T>&, Mat<T>&) const;
        void query(const Mat<T>&, Mat<T>&, Computer&) const;

        void query(const Mat<T>&, Mat<T>&, Mat<T>&, const Layer<T>&);
        void query(const Mat<T>&, Mat<T>&, Mat<T>&, const Layer<T>&, Computer&);

        void error(const Mat<T>&, const Mat<T>&, Mat<T>&) const;
        void error(const Mat<T>&, const Mat<T>&, Mat<T>&, Computer&) const;

        void error(const Mat<T>&, Mat<T>&, Mat<T>&, const Layer<T>&) const;
        void error(const Mat<T>&, Mat<T>&, Mat<T>&, const Layer<T>&, Computer&) const;

        T cost(Mat<T>&, const std::function<T(const T&)>&) const;

        void grad(Mat<T>&, const Mat<T>&, Mat<T>&, const std::function<T(const T&)>&) const;
        void grad(Mat<T>&, const Mat<T>&, Mat<T>&, const std::string&, Computer&) const;

        void train(Mat<T>&, const Layer<T>&, T);
        void train(Mat<T>&, const Layer<T>&, T, Computer& video);

        // High-level methods
    };
}

namespace ncf{
    namespace activation {
		template<typename T>
		T relu(const T& v){
			return v > 0 ? v : 0;
		}

		template<typename T>
		T lrelu(const T& v){
			return v > 0 ? v : v * T(0.1);
		}
	}
	namespace cost {
		template<typename T>
		T mse(const T& v) {
			return v * v;
		}
	}

	namespace derivative {
		namespace activation {
			template<typename T>
			T relu(const T& v) {
				return v > 0 ? 1 : 0;
			}

			template<typename T>
			T lrelu(const T& v) {
				return v > 0 ? 1 : T(0.1);
			}
		}
		namespace cost {
			template<typename T>
			T mse(const T& v) {
				return 2 * v;
			}
		}
	}

    namespace optimizer{
        template<typename T>
        void gd(Mat<T>& X, Mat<T>& grad, T learning_rate){
            grad.mul(learning_rate, grad);
            X.sub(grad, X);
        }
        template<typename T>
        void gd(Mat<T>& X, Mat<T>& grad, T learning_rate, ecl::Computer& video){
            grad.mul(learning_rate, grad, video);
            X.sub(grad, X, video);
        }
    }
}

// IMPLEMENTATION
// Layer
template<typename T>
ncf::Layer<T>::Layer(std::size_t neurons){
    this->neurons = neurons;
}

template<typename T>
void ncf::Layer<T>::send(ecl::Computer& video){
    for(auto& p : core) video << p.second;
}
template<typename T>
void ncf::Layer<T>::receive(ecl::Computer& video){
    for(auto& p : core) video >> p.second;
}
template<typename T>
void ncf::Layer<T>::grab(ecl::Computer& video){
    for(auto& p : core) p.second.grab(video);
}
template<typename T>
void ncf::Layer<T>::release(ecl::Computer& video){
    for(auto& p : core) p.second.release(video);
}

namespace ncf{
    template<typename T>
    Computer& operator<<(Computer& video, Layer<T>& other){
        other.send(video);
        return video;
    }
    template<typename T>
    Computer& operator>>(Computer& video, Layer<T>& other){
        other.receive(video);
        return video;
    }
}

template<typename T>
bool ncf::Layer<T>::checkCore(std::size_t prev_neurons) const{
    if(core.find(prev_neurons) == core.end()) return false;
    return true;
}
template<typename T>
void ncf::Layer<T>::createCore(std::size_t prev_neurons){
    if(!checkCore(prev_neurons)){
        if(coregen == nullptr)
            throw std::runtime_error("Layer [create core]: coregen method unsetted");

        Mat<T> new_core(neurons, prev_neurons);
        coregen(new_core);
        core.emplace(prev_neurons, std::move(new_core));
    }
}
template<typename T>
void ncf::Layer<T>::releaseCore(std::size_t prev_neurons){
    auto it = core.find(prev_neurons);
    if(it != core.end()){
        core.erase(it);
    }
}

template<typename T>
void ncf::Layer<T>::setActivation(const std::function<T(const T&)>& activation){
    this->activation = activation;
}
template<typename T>
void ncf::Layer<T>::setDerivative(const std::function<T(const T&)>& derivative){
    this->derivative = derivative;
}

template<typename T>
void ncf::Layer<T>::setActivation(const std::string& activation){
    this->computer_activation = activation;
}
template<typename T>
void ncf::Layer<T>::setDerivative(const std::string& derivative){
    this->computer_derivative = derivative;
}

template<typename T>
void ncf::Layer<T>::setCoreGen(const std::function<void(mcf::Mat<T>&)>& coregen){
    this->coregen = coregen;
}

template<typename T>
std::size_t ncf::Layer<T>::getNeurons() const{
    return neurons;
}
template<typename T>
mcf::Mat<T>& ncf::Layer<T>::getCore(std::size_t prev_neurons){
    return core.at(prev_neurons);
}
template<typename T>
const mcf::Mat<T>& ncf::Layer<T>::getConstCore(std::size_t prev_neurons) const{
    return core.at(prev_neurons);
}
template<typename T>
const std::string& ncf::Layer<T>::getComputerActivation() const{
    return computer_activation;
}
template<typename T>
const std::string& ncf::Layer<T>::getComputerDerivative() const{
    return computer_derivative;
}
template<typename T>
const std::function<void(mcf::Mat<T>&)>& ncf::Layer<T>::getCoreGen() const{
    return coregen;
}
template<typename T>
const std::function<T(const T&)>& ncf::Layer<T>::getActivation() const{
    return activation;
}
template<typename T>
const std::function<T(const T&)>& ncf::Layer<T>::getDerivative() const{
    return derivative;
}

template<typename T>
void ncf::Layer<T>::query(const mcf::Mat<T>& in, mcf::Mat<T>& out) const{
    if(activation == nullptr)
        throw std::runtime_error("Layer [query]: activation function unsetted");
    in.map(activation, out);
}
template<typename T>
void ncf::Layer<T>::query(const mcf::Mat<T>& in, mcf::Mat<T>& out, ecl::Computer& video) const{
    in.map(computer_activation, out, video);
}

template<typename T>
void ncf::Layer<T>::query(const mcf::Mat<T>& in, mcf::Mat<T>& preout, mcf::Mat<T>& out, const Layer<T>& prev){
    createCore(prev.neurons);
    
    core.at(prev.neurons).mul(in, preout);
    preout.map(activation, out);
}
template<typename T>
void ncf::Layer<T>::query(const mcf::Mat<T>& in, mcf::Mat<T>& preout, mcf::Mat<T>& out, const Layer<T>& prev, ecl::Computer& video){
    if(!checkCore(prev.neurons)){
        createCore(prev.neurons);
        send(video);
    }
    
    core.at(prev.neurons).mul(in, preout, video);
    preout.map(computer_activation, out, video);
}

template<typename T>
void ncf::Layer<T>::error(const mcf::Mat<T>& answer, const mcf::Mat<T>& out, mcf::Mat<T>& error) const{
    answer.sub(out, error);
}
template<typename T>
void ncf::Layer<T>::error(const mcf::Mat<T>& answer, const mcf::Mat<T>& out, mcf::Mat<T>& error, ecl::Computer& video) const{
    answer.sub(out, error, video);
}

template<typename T>
void ncf::Layer<T>::error(const mcf::Mat<T>& next_error, mcf::Mat<T>& preout, mcf::Mat<T>& error, const Layer<T>& next) const{
    if(derivative == nullptr)
        throw std::runtime_error("Layer [query]: derivative function unsetted");

    next.getConstCore(neurons).mul(next_error, error, ncf::TRANSPOSE::FIRST);
    preout.map(derivative, preout);
    error.hadamard(preout, error);
}
template<typename T>
void ncf::Layer<T>::error(const mcf::Mat<T>& next_error, mcf::Mat<T>& preout, mcf::Mat<T>& error, const Layer<T>& next, ecl::Computer& video) const{
    next.getConstCore(neurons).mul(next_error, error, video, ncf::TRANSPOSE::FIRST);
    preout.map(computer_derivative, preout, video);
    error.hadamard(preout, error, video);
}

template<typename T>
T ncf::Layer<T>::cost(mcf::Mat<T>& error, const std::function<T(const T&)>& cost) const{
    size_t count = error.getH() * error.getW();
    error.map(cost, error);
    return error.reduce() / static_cast<T>(count);
}

template<typename T>
void ncf::Layer<T>::grad(mcf::Mat<T>& error, const mcf::Mat<T>& prev_out, mcf::Mat<T>& grad, const std::function<T(const T&)>& div_cost) const{
    size_t count = error.getW() * error.getH();

    error.map(div_cost, error);
    error.mul(prev_out, grad, mcf::TRANSPOSE::SECOND);
    grad.map([&](const T& v){
        return -v / static_cast<T>(count);
    }, grad);
}
template<typename T>
void ncf::Layer<T>::grad(mcf::Mat<T>& error, const mcf::Mat<T>& prev_out, mcf::Mat<T>& grad, const std::string& div_cost, ecl::Computer& video) const{
    std::string count = std::to_string(error.getW() * error.getH());

    error.map(div_cost, error, video);
    error.mul(prev_out, grad, video, mcf::TRANSPOSE::SECOND);
    grad.map("ret = -v / " + count + ";", grad, video);
}

template<typename T>
void ncf::Layer<T>::train(mcf::Mat<T>& grad, const Layer<T>& prev, T learning_rate){
    createCore(prev.neurons);
    optimizer::gd<T>(core.at(prev.neurons), grad, learning_rate);
}
template<typename T>
void ncf::Layer<T>::train(mcf::Mat<T>& grad, const Layer<T>& prev, T learning_rate, ecl::Computer& video){
    if(!checkCore(prev.neurons)){
        createCore(prev.neurons);
        send(video);
    }
    optimizer::gd<T>(core.at(prev.neurons), grad, learning_rate, video);
}