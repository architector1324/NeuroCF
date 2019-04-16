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
        const Mat<T>& getCore(std::size_t) const;
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

        // High-level methods
    };
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
void ncf::Layer<T>::setCoreGen(const std::function<void(Mat<T>&)>& coregen){
    this->coregen = coregen;
}

template<typename T>
std::size_t ncf::Layer<T>::getNeurons() const{
    return neurons;
}
template<typename T>
const mcf::Mat<T>& ncf::Layer<T>::getCore(std::size_t prev_neurons) const{
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
void ncf::Layer<T>::query(const Mat<T>& in, Mat<T>& out) const{
    if(activation == nullptr)
        throw std::runtime_error("Layer [query]: activation function unsetted");
    in.map(activation, out);
}
template<typename T>
void ncf::Layer<T>::query(const Mat<T>& in, Mat<T>& out, Computer& video) const{
    in.map(computer_activation, out, video);
}

template<typename T>
void ncf::Layer<T>::query(const Mat<T>& in, Mat<T>& preout, Mat<T>& out, const Layer<T>& prev){
    createCore(prev.neurons);
    
    core.at(prev.neurons).mul(in, preout);
    preout.map(activation, out);
}
template<typename T>
void ncf::Layer<T>::query(const Mat<T>& in, Mat<T>& preout, Mat<T>& out, const Layer<T>& prev, Computer& video){
    if(!checkCore(prev.neurons)){
        createCore(prev.neurons);
        send(video);
    }
    
    core.at(prev.neurons).mul(in, preout, video);
    preout.map(computer_activation, out, video);
}

template<typename T>
void ncf::Layer<T>::error(const Mat<T>& answer, const Mat<T>& out, Mat<T>& error) const{
    answer.sub(out, error);
}
template<typename T>
void ncf::Layer<T>::error(const Mat<T>& answer, const Mat<T>& out, Mat<T>& error, Computer& video) const{
    answer.sub(out, error, video);
}

template<typename T>
void ncf::Layer<T>::error(const Mat<T>& next_error, Mat<T>& preout, Mat<T>& error, const Layer<T>& next) const{
    if(derivative == nullptr)
        throw std::runtime_error("Layer [query]: derivative function unsetted");

    next.getCore(neurons).mul(next_error, error, ncf::TRANSPOSE::FIRST);
    preout.map(derivative, preout);
    error.hadamard(preout, error);
}
template<typename T>
void ncf::Layer<T>::error(const Mat<T>& next_error, Mat<T>& preout, Mat<T>& error, const Layer<T>& next, Computer& video) const{
    next.getCore(neurons).mul(next_error, error, video, ncf::TRANSPOSE::FIRST);
    preout.map(computer_derivative, preout, video);
    error.hadamard(preout, error, video);
}

template<typename T>
T ncf::Layer<T>::cost(Mat<T>& error, const std::function<T(const T&)>& cost) const{
    size_t count = error.getH() * error.getW();
    error.map(cost, error);
    return error.reduce() / static_cast<T>(count);
}

template<typename T>
void ncf::Layer<T>::grad(Mat<T>& error, const Mat<T>& prev_out, Mat<T>& grad, const std::function<T(const T&)>& div_cost) const{
    size_t count = error.getW() * error.getH();

    error.map(div_cost, error);
    error.mul(prev_out, grad, mcf::TRANSPOSE::SECOND);
    grad.map([&](const T& v){
        return -v / static_cast<T>(count);
    }, grad);
}
template<typename T>
void ncf::Layer<T>::grad(Mat<T>& error, const Mat<T>& prev_out, Mat<T>& grad, const std::string& div_cost, Computer& video) const{
    std::string count = std::to_string(error.getW() * error.getH());

    error.map(div_cost, error, video);
    error.mul(prev_out, grad, video, mcf::TRANSPOSE::SECOND);
    grad.map("ret = -v / " + count + ";", grad, video);
}