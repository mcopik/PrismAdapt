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
package simulator.gpu.opencl.kernel.memory;

import prism.Preconditions;
import simulator.gpu.opencl.kernel.expression.Expression;

public class CLVariable implements CLValue
{
	public enum Location {
		REGISTER, LOCAL, GLOBAL
	}

	public final String varName;
	public final VariableInterface varType;
	private Expression initValue = null;
	private boolean dontDeclare = false;
	public Location memLocation = Location.REGISTER;

	public CLVariable(VariableInterface varType, String varName)
	{
		this.varName = varName;
		this.varType = varType;
	}

	public void setMemoryLocation(Location loc)
	{
		memLocation = loc;
	}

	public void dontDeclareVar()
	{
		dontDeclare = true;
	}

	public void setInitValue(CLValue value)
	{
		initValue = value.getSource();
	}

	public void setInitValue(Expression value)
	{
		initValue = value;
	}

	public VariableInterface getPointer()
	{
		if (varType.isArray()) {
			return new ArrayType((ArrayType) varType);
		} else {
			return new PointerType(varType);
		}
	}

	public CLVariable convertToPointer()
	{
		return new CLVariable(getPointer(), "&" + varName);
	}

	public CLVariable accessElement(Expression index)
	{
		return varType.accessElement(this, index);
	}

	public CLVariable accessField(String fieldName)
	{
		return varType.accessField(this.varName, fieldName);
	}

	public Expression getName()
	{
		return new Expression(varName);
	}

	public Expression cast(String type)
	{
		return new Expression(String.format("((%s)%s)", type, varName));
	}

	public CLVariable dereference()
	{
		Preconditions.checkCondition(varType instanceof PointerType || varType instanceof ArrayType, "Dereference on non-pointer!");
		if (varType instanceof PointerType) {
			return new CLVariable(((PointerType) varType).getInternalType(), (String.format("(*%s)", varName)));
		} else {
			return new CLVariable(((ArrayType) varType).getInternalType(), (String.format("(*%s)", varName)));
		}
	}

	public Expression getDeclaration()
	{
		if (dontDeclare) {
			return new Expression("");
		}
		StringBuilder builder = new StringBuilder();
		if (memLocation == Location.LOCAL) {
			builder.append("__local ");
		} else if (memLocation == Location.GLOBAL) {
			builder.append("__global ");
		}
		builder.append(varType.declareVar(varName));
		builder.append(";");
		return new Expression(builder.toString());
	}

	public Expression getDefinition()
	{
		if (dontDeclare) {
			return new Expression("");
		}
		StringBuilder builder = new StringBuilder();
		if (memLocation == Location.LOCAL) {
			builder.append("__local ");
		} else if (memLocation == Location.GLOBAL) {
			builder.append("__global ");
		}
		builder.append(varType.declareVar(varName));
		if (initValue != null) {
			builder.append(" = ").append(initValue.getSource());
		}
		builder.append(";");
		return new Expression(builder.toString());
	}

	@Override
	public boolean validateAssignmentTo(VariableInterface type)
	{
		//todo: implement
		return false;
	}

	@Override
	public Expression getSource()
	{
		return new Expression(varName);
	}
}