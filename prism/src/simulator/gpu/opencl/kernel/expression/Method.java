//==============================================================================
//	
//	Copyright (c) 2002-
//	Authors:
//	* Marcin Copik <mcopik@gmail.com> (Silesian University of Technology)
//	
//------------------------------------------------------------------------------
//	
//	This file is part of PRISM.
//	
//	PRISM is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//	
//	PRISM is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//	
//	You should have received a copy of the GNU General Public License
//	along with PRISM; if not, write to the Free Software Foundation,
//	Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//	
//==============================================================================
package simulator.gpu.opencl.kernel.expression;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;

import prism.Preconditions;
import simulator.gpu.opencl.kernel.KernelException;
import simulator.gpu.opencl.kernel.memory.CLValue;
import simulator.gpu.opencl.kernel.memory.CLVariable;
import simulator.gpu.opencl.kernel.memory.VariableInterface;

public class Method extends ComplexKernelComponent
{
	public final String methodName;
	public final VariableInterface methodType;
	protected Map<String, CLVariable> args = new LinkedHashMap<>();
	protected CLVariable stateVectorAccess = null;

	//protected CLVariable. returnType = new CLVariable(CLVariable.Type.VOID);

	public Method(String name, VariableInterface returnType)
	{
		methodName = name;
		methodType = returnType;
	}

	public void addArg(CLVariable var) throws KernelException
	{
		if (args.containsKey(var.varName)) {
			throw new KernelException("Variable " + var.varName + " already exists in arg list of method " + methodName);
		}
		args.put(var.varName, var);
		updateIncludes(var);
	}

	public void addArg(Collection<CLVariable> vars) throws KernelException
	{
		for (CLVariable var : vars) {
			addArg(var);
		}
	}

	public int getArgsSize()
	{
		return args.size();
	}

	public CLVariable getArg(String name)
	{
		return args.get(name);
	}

	public int getVarsNum()
	{
		return localVars.size();
	}

	public void registerStateVector(CLVariable var)
	{
		Preconditions.checkCondition(args.containsValue(var) || localVars.containsValue(var),
		//global variables do not exist in OpenCL - it has to be arg/local var
				"StateVector reference has to be a local variable or function argument!");
		stateVectorAccess = var;
	}

	@Override
	protected String createHeader()
	{
		StringBuilder builder = new StringBuilder(methodType.getType());
		builder.append(" ").append(methodName).append("( ");
		for (Map.Entry<String, CLVariable> decl : args.entrySet()) {
			CLVariable var = decl.getValue();
			builder.append(var.getDeclaration());
			builder.deleteCharAt(builder.length() - 1);
			builder.append(", ");
		}
		if (args.size() != 0) {
			int len = builder.length();
			builder.replace(len - 2, len - 1, ")");
		} else {
			builder.append(")");
		}
		return builder.toString();
	}

	@Override
	public KernelComponent getDeclaration()
	{
		return new Expression(createHeader() + ";");
	}

	public Expression callMethod(CLValue... args)
	{
		StringBuilder builder = new StringBuilder(methodName);
		builder.append("(");
		for (CLValue arg : args) {
			builder.append(arg.getSource()).append(",");
		}
		builder.deleteCharAt(builder.length() - 1);
		builder.append(")");
		return new Expression(builder.toString());
	}

	@Override
	public boolean hasDeclaration()
	{
		return true;
	}

	@Override
	public void accept(VisitorInterface v)
	{
		for (Expression expr : variableDefinitions) {
			expr.accept(v);
		}
		for (KernelComponent component : body) {
			component.accept(v);
		}
	}

	public void addReturn(CLVariable var)
	{
		body.add(new Expression(String.format("return %s;", var.varName)));
	}

	public void addReturn(Expression expr)
	{
		body.add(new Expression(String.format("return %s;", expr)));
	}

	public boolean hasDefinedSVAccess()
	{
		return stateVectorAccess != null;
	}

	public CLVariable accessStateVector()
	{
		return stateVectorAccess;
	}
}