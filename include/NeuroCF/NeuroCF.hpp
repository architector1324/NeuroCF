#pragma once

#include "MatrixCF.hpp"

namespace ncf{
    using namespace mcf;
    using namespace ecl;

    template<typename T>
    class Layer{
    private:
        std::map<std::size_t, Mat<T>> core;
        size_t neurons = 0;
        std::function<T(const T&)> activation = nullptr;
        std::string computer_activation = "";

    public:
        Layer() = delete;
        explicit Layer(std::size_t);

        void setActivation(const std::function<T(const T&)>&);
        void setActivation(const std::string&);

        void query(const Mat<T>&, Mat<T>&) const;
        void query(const Mat<T>&, Mat<T>&, Computer&) const;

        void send(Computer&);
        void receive(Computer&);
        void grab(Computer&);
        void release(Computer&);

        template<typename U>
        friend Computer& operator<<(Computer&, Layer<U>&);
        template<typename U>
        friend Computer& operator>>(Computer&, Layer<U>&);
    };
}

// IMPLEMENTATION
// Layer
template<typename T>
ncf::Layer<T>::Layer(std::size_t neurons){
    this->neurons = neurons;
}

template<typename T>
void ncf::Layer<T>::setActivation(const std::function<T(const T&)>& activation){
    this->activation = activation;
}
template<typename T>
void ncf::Layer<T>::setActivation(const std::string& activation){
    this->computer_activation = activation;
}

template<typename T>
void ncf::Layer<T>::query(const Mat<T>& in, Mat<T>& out) const{
    if(activation == nullptr) throw std::runtime_error("Layer [query]: activation function unsetted");
    in.map(activation, out);
}
template<typename T>
void ncf::Layer<T>::query(const Mat<T>& in, Mat<T>& out, ecl::Computer& video) const{
    in.map(computer_activation, out, video);
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